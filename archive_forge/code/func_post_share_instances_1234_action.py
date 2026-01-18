import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_instances_1234_action(self, body, **kw):
    _body = None
    resp = 202
    assert len(list(body)) == 1
    action = list(body)[0]
    if action in ('reset_status', 'os-reset_status'):
        assert 'status' in body.get('reset_status', body.get('os-reset_status'))
    elif action == 'os-force_delete':
        assert body[action] is None
    else:
        raise AssertionError('Unexpected share action: %s' % action)
    return (resp, {}, _body)