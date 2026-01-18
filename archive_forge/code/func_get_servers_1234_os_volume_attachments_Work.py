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
def get_servers_1234_os_volume_attachments_Work(self, **kw):
    return (200, FAKE_RESPONSE_HEADERS, {'volumeAttachment': {'display_name': 'Work', 'display_description': 'volume for work', 'status': 'ATTACHED', 'id': '15e59938-07d5-11e1-90e3-e3dffe0c5983', 'created_at': '2011-09-09T00:00:00Z', 'attached': '2011-11-11T00:00:00Z', 'size': 1024, 'attachments': [{'id': '3333', 'links': ''}], 'metadata': {}}})