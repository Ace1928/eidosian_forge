from unittest import mock
import uuid
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import limit as klimit
from openstack.identity.v3 import registered_limit
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
def test_get_limit(self):
    utils = limit._EnforcerUtils(cache=False)
    mgpl = mock.MagicMock()
    mgrl = mock.MagicMock()
    with mock.patch.multiple(utils, _get_project_limit=mgpl, _get_registered_limit=mgrl):
        utils._get_limit('project', 'foo')
        mgrl.assert_not_called()
        mgpl.assert_called_once_with('project', 'foo')
        mgrl.reset_mock()
        mgpl.reset_mock()
        mgpl.return_value = None
        utils._get_limit('project', 'foo')
        mgrl.assert_called_once_with('foo')
        mgpl.assert_called_once_with('project', 'foo')
        mgrl.reset_mock()
        mgpl.reset_mock()
        utils._get_limit(None, 'foo')
        mgrl.assert_called_once_with('foo')
        mgpl.assert_not_called()