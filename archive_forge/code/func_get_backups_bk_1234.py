from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_backups_bk_1234(self, **kw):
    r = {'backup': self.get_backups()[2]['backups'][0]}
    return (200, {}, r)