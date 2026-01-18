import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_shares_1111_action(self, body, **kw):
    _body = None
    resp = 202
    assert len(list(body)) == 1
    action = list(body)[0]
    if action in ('allow_access', 'os-allow_access'):
        expected = ['access_level', 'access_to', 'access_type']
        actual = sorted(list(body[action]))
        err_msg = "expected '%s', actual is '%s'" % (expected, actual)
        assert expected == actual, err_msg
        _body = {'access': {}}
    elif action in ('access_list', 'os-access_list'):
        assert body[action] is None
        _body = {'access_list': [{'access_level': 'rw', 'state': 'active', 'id': '1122', 'access_type': 'ip', 'access_to': '10.0.0.7'}]}
    else:
        raise AssertionError('Unexpected share action: %s' % action)
    return (resp, {}, _body)