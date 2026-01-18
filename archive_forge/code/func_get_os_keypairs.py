from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_os_keypairs(self, *kw):
    return (200, {'keypairs': [{'fingerprint': 'FAKE_KEYPAIR', 'name': 'test', 'public_key': 'foo'}]})