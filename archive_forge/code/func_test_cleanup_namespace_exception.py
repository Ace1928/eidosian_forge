import datetime
from unittest import mock
from oslo_serialization import jsonutils
import webob
import wsme
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
from glance.api.v2 import metadef_objects as objects
from glance.api.v2 import metadef_properties as properties
from glance.api.v2 import metadef_resource_types as resource_types
from glance.api.v2 import metadef_tags as tags
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
@mock.patch('glance.api.v2.metadef_namespaces.LOG')
@mock.patch('glance.notifier.MetadefNamespaceRepoProxy.remove')
def test_cleanup_namespace_exception(self, mock_remove, mock_log):
    mock_remove.side_effect = Exception('Mock remove was called')
    fake_gateway = glance.gateway.Gateway(db_api=self.db, notifier=self.notifier, policy_enforcer=self.policy)
    req = unit_test_utils.get_fake_request(roles=['admin'])
    namespace = namespaces.Namespace()
    namespace.namespace = 'FakeNamespace'
    namespace = self.namespace_controller.create(req, namespace)
    ns_repo = fake_gateway.get_metadef_namespace_repo(req.context)
    self.namespace_controller._cleanup_namespace(ns_repo, namespace, True)
    called_msg = 'Failed to delete namespace %(namespace)s.Exception: %(exception)s'
    called_args = {'exception': 'Mock remove was called', 'namespace': 'FakeNamespace'}
    mock_log.error.assert_called_with((called_msg, called_args))
    mock_remove.assert_called_once_with(mock.ANY)