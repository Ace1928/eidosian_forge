import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def test__authenticate(self):
    authObj = auth.Authenticator(mock.Mock(), auth.KeyStoneV2Authenticator, mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock())
    resp = mock.Mock()
    resp.status = 200
    body = 'test_body'
    auth.ServiceCatalog._load = mock.Mock(return_value=1)
    authObj.client._time_request = mock.Mock(return_value=(resp, body))
    sc = authObj._authenticate(mock.Mock(), mock.Mock())
    self.assertEqual(body, sc.catalog)
    auth.ServiceCatalog.__init__ = mock.Mock(side_effect=exceptions.AmbiguousEndpoints)
    self.assertRaises(exceptions.AmbiguousEndpoints, authObj._authenticate, mock.Mock(), mock.Mock())
    auth.ServiceCatalog.__init__ = mock.Mock(side_effect=KeyError)
    self.assertRaises(exceptions.AuthorizationFailure, authObj._authenticate, mock.Mock(), mock.Mock())
    mock_obj = mock.Mock(side_effect=exceptions.EndpointNotFound)
    auth.ServiceCatalog.__init__ = mock_obj
    self.assertRaises(exceptions.EndpointNotFound, authObj._authenticate, mock.Mock(), mock.Mock())
    mock_obj.side_effect = None
    resp.__getitem__ = mock.Mock(return_value='loc')
    resp.status = 305
    body = 'test_body'
    authObj.client._time_request = mock.Mock(return_value=(resp, body))
    lo = authObj._authenticate(mock.Mock(), mock.Mock())
    self.assertEqual('loc', lo)
    resp.status = 404
    exceptions.from_response = mock.Mock(side_effect=ValueError)
    self.assertRaises(ValueError, authObj._authenticate, mock.Mock(), mock.Mock())