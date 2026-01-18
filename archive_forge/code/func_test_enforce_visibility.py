from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
@mock.patch('glance.api.policy._enforce_image_visibility')
def test_enforce_visibility(self, mock_enf):
    self.policy._enforce_visibility('something')
    mock_enf.assert_called_once_with(self.enforcer, self.context, 'something', mock.ANY)
    mock_enf.side_effect = exception.Forbidden
    self.assertRaises(webob.exc.HTTPForbidden, self.policy._enforce_visibility, 'something')
    mock_enf.side_effect = exception.ImageNotFound
    self.assertRaises(exception.ImageNotFound, self.policy._enforce_visibility, 'something')