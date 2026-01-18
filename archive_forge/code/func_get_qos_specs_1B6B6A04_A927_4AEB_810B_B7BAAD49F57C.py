from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_qos_specs_1B6B6A04_A927_4AEB_810B_B7BAAD49F57C(self, **kw):
    base_uri = 'http://localhost:8776'
    tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
    qos_id1 = '1B6B6A04-A927-4AEB-810B-B7BAAD49F57C'
    return (200, {}, _stub_qos_full(qos_id1, base_uri, tenant_id))