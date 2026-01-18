from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_os_networks(self, **kw):
    return (200, {'networks': [{'label': 'public', 'id': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'}, {'label': 'foo', 'id': '42'}, {'label': 'foo', 'id': '42'}]})