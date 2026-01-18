from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_security_groups_2(self, **kw):
    r = {'security_group': self.get_security_groups()[2]['security_groups'][0]}
    return (200, {}, r)