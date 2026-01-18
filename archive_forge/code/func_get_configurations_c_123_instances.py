from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_configurations_c_123_instances(self, **kw):
    return (200, {}, {'instances': [{'id': '1', 'name': 'instance-1'}, {'id': '2', 'name': 'instance-2'}]})