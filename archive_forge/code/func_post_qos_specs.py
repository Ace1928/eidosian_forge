from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_qos_specs(self, **kw):
    base_uri = 'http://localhost:8776'
    tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
    qos_id = '1B6B6A04-A927-4AEB-810B-B7BAAD49F57C'
    qos_name = 'qos-name'
    return (202, {}, _stub_qos_full(qos_id, base_uri, tenant_id, qos_name))