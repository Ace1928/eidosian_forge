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
def test_create_object_index_rax(self):
    self.register_uris([dict(method='PUT', uri='{endpoint}/{container}/{object}'.format(endpoint=self.endpoint, container=self.container, object='index.html'), status_code=201, validate=dict(headers={'access-control-allow-origin': '*', 'content-type': 'text/html'}))])
    headers = {'access-control-allow-origin': '*', 'content-type': 'text/html'}
    self.cloud.create_object(self.container, name='index.html', data='', **headers)
    self.assert_calls()