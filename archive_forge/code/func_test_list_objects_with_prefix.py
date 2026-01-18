import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils
def test_list_objects_with_prefix(self):
    endpoint = '{endpoint}?format=json&prefix=test'.format(endpoint=self.container_endpoint)
    objects = [{u'bytes': 20304400896, u'last_modified': u'2016-12-15T13:34:13.650090', u'hash': u'daaf9ed2106d09bba96cf193d866445e', u'name': self.object, u'content_type': u'application/octet-stream'}]
    self.register_uris([dict(method='GET', uri=endpoint, complete_qs=True, json=objects)])
    ret = self.cloud.list_objects(self.container, prefix='test')
    self.assert_calls()
    for a, b in zip(objects, ret):
        self._compare_objects(a, b)