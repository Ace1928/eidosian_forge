from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def post_servers_1234_action(self, body, **kw):
    _body = None
    resp = 202
    if len(body.keys()) != 1:
        raise AssertionError('No keys in request body')
    action = next(iter(body))
    keys = list(body[action].keys()) if body[action] is not None else None
    if action == 'reboot':
        if keys != ['type']:
            raise AssertionError('Unexpection action keys for %s: %s' % (action, keys))
        if body[action]['type'] not in ['HARD', 'SOFT']:
            raise AssertionError('Unexpected reboot type %s' % body[action]['type'])
    elif action == 'rebuild':
        if 'adminPass' in keys:
            keys.remove('adminPass')
        if keys != ['imageRef']:
            raise AssertionError('Unexpection action keys for %s: %s' % (action, keys))
        _body = self.get_servers_1234()[1]
    elif action == 'confirmResize':
        if body[action] is not None:
            raise AssertionError('Unexpected data for confirmResize: %s' % body[action])
        return (204, None)
    elif action in ['revertResize', 'migrate', 'rescue', 'unrescue', 'suspend', 'resume', 'lock', 'unlock', 'forceDelete']:
        if body[action] is not None:
            raise AssertionError('Unexpected data for %s: %s' % (action, body[action]))
    else:
        expected_keys = {'resize': {'flavorRef'}, 'addFixedIp': {'networkId'}, 'removeFixedIp': {'address'}, 'addFloatingIp': {'address'}, 'removeFloatingp': {'address'}, 'createImage': {'name', 'metadata'}, 'changePassword': {'adminPass'}, 'os-getConsoleOutput': {'length'}, 'os-getVNCConsole': {'type'}, 'os-migrateLive': {'host', 'block_migration', 'disk_over_commit'}}
        if action in expected_keys:
            if set(keys) != set(expected_keys[action]):
                raise AssertionError('Unexpection action keys for %s: %s' % (action, keys))
        else:
            raise AssertionError('Unexpected server action: %s' % action)
        if action == 'createImage':
            resp = {'status': 202, 'location': 'http://blah/images/456'}
        if action == 'os-getConsoleOutput':
            return (202, {'output': 'foo'})
    return (resp, _body)