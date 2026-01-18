from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def post_clusters_cls_1234_root(self, **kw):
    return (202, {}, {'user': {'password': 'password', 'name': 'root'}})