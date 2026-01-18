import copy
import datetime
import re
from unittest import mock
from urllib import parse
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client as base_client
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils
from novaclient.v2 import client
def get_servers_9013(self, **kw):
    r = {'server': self.get_servers_detail()[2]['servers'][3]}
    return (200, {}, r)