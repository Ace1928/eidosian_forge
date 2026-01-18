import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_servers_1234_action(self, body, **kw):
    _body = None
    assert len(list(body)) == 1
    action = list(body)[0]
    if action in ('reset_status',):
        assert 'status' in body.get('reset_status', body.get('os-reset_status'))
        _body = {'reset_status': {'status': body['reset_status']['status']}}
    elif action in ('unmanage',):
        assert 'force' in body[action]
    elif action in ('migration_cancel', 'migration_complete', 'migration_get_progress'):
        assert body[action] is None
        if 'migration_get_progress' == action:
            _body = {'total_progress': 50, 'task_state': 'fake_task_state', 'destination_share_server_id': 'fake_dest_id'}
            return (200, {}, _body)
        elif 'migration_complete' == action:
            _body = {'destination_share_server_id': 'fake_dest_id'}
            return (200, {}, _body)
    elif action in ('migration_start', 'migration_check'):
        assert 'host' in body[action]
        if 'migration-check':
            _body = {'compatible': True, 'capacity': True, 'capability': True, 'writable': True, 'nondisruptive': True, 'preserve_snapshots': True}
            return (200, {}, _body)
    elif action == 'reset_task_state':
        assert 'task_state' in body[action]
    resp = 202
    result = (resp, {}, _body)
    return result