from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_flavors_3(self, **kw):
    return (200, {'flavor': {'id': 3, 'name': 'm1.large', 'ram': 512, 'disk': 20, 'OS-FLV-EXT-DATA:ephemeral': 30}})