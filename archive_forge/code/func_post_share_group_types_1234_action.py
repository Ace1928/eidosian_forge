import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_group_types_1234_action(self, body, **kw):
    assert len(list(body)) == 1
    action = list(body)[0]
    if action == 'addProjectAccess':
        assert 'project' in body['addProjectAccess']
    elif action == 'removeProjectAccess':
        assert 'project' in body['removeProjectAccess']
    else:
        raise AssertionError('Unexpected action: %s' % action)
    return (202, {}, None)