from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_instances_master_1(self, **kw):
    return (200, {}, {'instance': {'id': 'myid'}})