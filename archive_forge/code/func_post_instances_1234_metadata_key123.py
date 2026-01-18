from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def post_instances_1234_metadata_key123(self, body, **kw):
    return (202, {}, {'metadata': {}})