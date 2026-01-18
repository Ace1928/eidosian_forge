from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_flavors_eph_rd_smaller(self, **kw):
    r = {'flavor': self.get_flavors()[2]['flavors'][1]}
    return (200, {}, r)