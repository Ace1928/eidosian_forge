from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def post_instances_1234_databases(self, body, **kw):
    assert_has_keys(body, required=['databases'])
    for database in body['databases']:
        assert_has_keys(database, required=['name'], optional=['character_set', 'collate'])
    return (202, {}, self.get_instances_1234_databases()[2]['databases'][0])