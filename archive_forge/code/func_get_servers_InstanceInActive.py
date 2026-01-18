from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_servers_InstanceInActive(self, **kw):
    r = {'server': self.get_servers_detail()[1]['servers'][8]}
    return (200, r)