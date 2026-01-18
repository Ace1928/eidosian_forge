from unittest import mock
import ddt
from oslo_utils import uuidutils
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
def fake_url_for(version):
    if version == 'v3.0':
        return 'url_v3.0'
    elif version == 'v2.0' and self.auth_url == 'both':
        return 'url_v2.0'
    else:
        return None