import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_replicas_1234_action(self, body, **kw):
    _body = None
    resp = 202
    assert len(list(body)) == 1
    action = list(body)[0]
    if action in ('reset_status', 'reset_replica_state'):
        attr = action.split('reset_')[1]
        assert attr in body.get(action)
    elif action in ('force_delete', 'resync'):
        assert body[action] is None
    elif action not in 'promote':
        raise AssertionError('Unexpected share action: %s' % action)
    return (resp, {}, _body)