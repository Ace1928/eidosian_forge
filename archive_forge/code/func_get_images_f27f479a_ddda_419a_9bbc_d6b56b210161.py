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
def get_images_f27f479a_ddda_419a_9bbc_d6b56b210161(self, **kw):
    return (200, {}, {'image': self.get_images()[2]['images'][3]})