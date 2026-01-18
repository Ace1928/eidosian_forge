from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import client
from manilaclient import exceptions
from manilaclient.tests.unit import utils
import manilaclient.v1.client
import manilaclient.v2.client
@ddt.data(('1', '1.0'), ('1', '2.0'), ('1', '2.7'), ('1', None), ('1.0', '1.0'), ('1.0', '2.0'), ('1.0', '2.7'), ('1.0', None), ('2', '1.0'), ('2', '2.0'), ('2', '2.7'), ('2', None))
@ddt.unpack
def test_init_client_with_version_parms(self, pos, kw):
    major = int(float(pos))
    pos_av = mock.Mock()
    kw_av = mock.Mock()
    with mock.patch.object(manilaclient.v1.client, 'Client'):
        with mock.patch.object(manilaclient.v2.client, 'Client'):
            with mock.patch.object(api_versions, 'APIVersion'):
                api_versions.APIVersion.side_effect = [pos_av, kw_av]
                pos_av.get_major_version.return_value = str(major)
                if kw is None:
                    manilaclient.client.Client(pos, 'foo')
                    expected_av = pos_av
                else:
                    manilaclient.client.Client(pos, 'foo', api_version=kw)
                    expected_av = kw_av
                if int(float(pos)) == 1:
                    expected_client_ver = api_versions.DEPRECATED_VERSION
                    self.assertFalse(manilaclient.v2.client.Client.called)
                    manilaclient.v1.client.Client.assert_has_calls([mock.call('foo', api_version=expected_av)])
                else:
                    expected_client_ver = api_versions.MIN_VERSION
                    self.assertFalse(manilaclient.v1.client.Client.called)
                    manilaclient.v2.client.Client.assert_has_calls([mock.call('foo', api_version=expected_av)])
                if kw is None:
                    api_versions.APIVersion.assert_called_once_with(expected_client_ver)
                else:
                    api_versions.APIVersion.assert_has_calls([mock.call(expected_client_ver), mock.call(kw)])