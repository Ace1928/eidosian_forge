from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_version_map():
    return {'1.0': 'troveclient.tests.fakes.FakeClient'}