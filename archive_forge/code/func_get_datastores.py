from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_datastores(self, **kw):
    return (200, {}, {'datastores': [{'default_version': 'v-56', 'versions': [{'id': 'v-56', 'name': '5.6'}], 'id': 'd-123', 'name': 'mysql'}, {'default_version': 'v-71', 'versions': [{'id': 'v-71', 'name': '7.1'}], 'id': 'd-456', 'name': 'vertica'}]})