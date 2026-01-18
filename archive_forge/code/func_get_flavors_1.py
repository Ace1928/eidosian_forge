from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_flavors_1(self, **kw):
    return (200, {'flavor': {'id': 1, 'name': '256 MB Server', 'ram': 256, 'disk': 10, 'OS-FLV-EXT-DATA:ephemeral': 10}})