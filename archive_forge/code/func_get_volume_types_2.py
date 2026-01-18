from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_volume_types_2(self, **kw):
    r = {'volume_type': self.get_volume_types()[2]['volume_types'][2]}
    return (200, {}, r)