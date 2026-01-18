from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_instances_member_1_modules(self, **kw):
    return self.get_modules()