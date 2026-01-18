from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_os_floating_ips_1(self, **kw):
    return (200, {'floating_ip': {'id': 1, 'fixed_ip': '10.0.0.1', 'ip': '11.0.0.1'}})