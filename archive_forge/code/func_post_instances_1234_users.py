from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def post_instances_1234_users(self, body, **kw):
    assert_has_keys(body, required=['users'])
    for database in body['users']:
        assert_has_keys(database, required=['name', 'password'], optional=['databases'])
    return (202, {}, self.get_instances_1234_users()[2]['users'][0])