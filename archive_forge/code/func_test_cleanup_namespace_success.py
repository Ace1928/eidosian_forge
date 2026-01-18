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
def test_cleanup_namespace_success(self, mock_log):
    fake_gateway = glance.gateway.Gateway(db_api=self.db, notifier=self.notifier, policy_enforcer=self.policy)
    req = unit_test_utils.get_fake_request(roles=['admin'])
    namespace = namespaces.Namespace()
    namespace.namespace = 'FakeNamespace'
    namespace = self.namespace_controller.create(req, namespace)
    ns_repo = fake_gateway.get_metadef_namespace_repo(req.context)
    self.namespace_controller._cleanup_namespace(ns_repo, namespace, True)
    mock_log.debug.assert_called_with('Cleaned up namespace %(namespace)s ', {'namespace': namespace.namespace})