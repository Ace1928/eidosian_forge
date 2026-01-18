from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_volume_transfers_detail(self, **kw):
    base_uri = 'http://localhost:8776'
    tenant_id = '0fa851f6668144cf9cd8c8419c1646c1'
    transfer1 = '5678'
    transfer2 = 'f625ec3e-13dd-4498-a22a-50afd534cc41'
    return (200, {}, {'transfers': [fakes_base._stub_transfer_full(transfer1, base_uri, tenant_id), fakes_base._stub_transfer_full(transfer2, base_uri, tenant_id)]})