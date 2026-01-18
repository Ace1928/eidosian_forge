import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_shares_detail(self, **kw):
    endpoint = 'http://127.0.0.1:8786/v2'
    share_id = '1234'
    shares = {'shares': [{'id': share_id, 'name': 'sharename', 'status': 'fake_status', 'size': 1, 'host': 'fake_host', 'export_location': 'fake_export_location', 'snapshot_id': 'fake_snapshot_id', 'links': [{'href': endpoint + '/fake_project/shares/' + share_id, 'rel': 'self'}]}]}
    if kw.get('with_count'):
        shares.update({'count': 2})
    return (200, {}, shares)