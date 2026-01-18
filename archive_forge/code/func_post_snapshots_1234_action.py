from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_snapshots_1234_action(self, body, **kw):
    _body = None
    resp = 202
    assert len(list(body)) == 1
    action = list(body)[0]
    if action == 'os-reset_status':
        assert 'status' in body['os-reset_status']
    elif action == 'os-update_snapshot_status':
        assert 'status' in body['os-update_snapshot_status']
    elif action == 'os-force_delete':
        assert body[action] is None
    elif action == 'os-unmanage':
        assert body[action] is None
    else:
        raise AssertionError('Unexpected action: %s' % action)
    return (resp, {}, _body)