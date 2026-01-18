import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_adapter_opts_set(self):
    """Adapter opts specified in the conf."""
    conn = self._get_conn()
    discovery = {'versions': {'values': [{'status': 'stable', 'updated': '2019-06-01T00:00:00Z', 'media-types': [{'base': 'application/json', 'type': 'application/vnd.openstack.heat-v2+json'}], 'id': 'v2.0', 'links': [{'href': 'https://example.org:8888/heat/v2', 'rel': 'self'}]}]}}
    self.register_uris([dict(method='GET', uri='https://example.org:8888/heat/v2', json=discovery), dict(method='GET', uri='https://example.org:8888/heat/v2/foo', json={'foo': {}})])
    adap = conn.orchestration
    self.assertEqual('SpecialRegion', adap.region_name)
    self.assertEqual('orchestration', adap.service_type)
    self.assertEqual('internal', adap.interface)
    self.assertEqual('https://example.org:8888/heat/v2', adap.endpoint_override)
    adap.get('/foo')
    self.assert_calls()