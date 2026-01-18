import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
def event_list_resp_dict(self, stack_name='teststack', resource_name=None, rsrc_eventid1='7fecaeed-d237-4559-93a5-92d5d9111205', rsrc_eventid2='e953547a-18f8-40a7-8e63-4ec4f509648b', final_state='COMPLETE'):
    action = 'CREATE'
    rn = resource_name if resource_name else 'testresource'
    resp_dict = {'events': [{'event_time': '2013-12-05T14:14:31', 'id': rsrc_eventid1, 'links': [{'href': 'http://heat.example.com:8004/foo', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo2', 'rel': 'resource'}, {'href': 'http://heat.example.com:8004/foo3', 'rel': 'stack'}], 'logical_resource_id': 'myDeployment', 'physical_resource_id': None, 'resource_name': rn, 'resource_status': '%s_IN_PROGRESS' % action, 'resource_status_reason': 'state changed'}, {'event_time': '2013-12-05T14:14:32', 'id': rsrc_eventid2, 'links': [{'href': 'http://heat.example.com:8004/foo', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo2', 'rel': 'resource'}, {'href': 'http://heat.example.com:8004/foo3', 'rel': 'stack'}], 'logical_resource_id': 'myDeployment', 'physical_resource_id': 'bce15ec4-8919-4a02-8a90-680960fb3731', 'resource_name': rn, 'resource_status': '%s_%s' % (action, final_state), 'resource_status_reason': 'state changed'}]}
    if resource_name is None:
        stack_event1 = '0159dccd-65e1-46e8-a094-697d20b009e5'
        stack_event2 = '8f591a36-7190-4adb-80da-00191fe22388'
        resp_dict['events'].insert(0, {'event_time': '2013-12-05T14:14:30', 'id': stack_event1, 'links': [{'href': 'http://heat.example.com:8004/foo', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo2', 'rel': 'resource'}, {'href': 'http://heat.example.com:8004/foo3', 'rel': 'stack'}], 'logical_resource_id': 'aResource', 'physical_resource_id': 'foo3', 'resource_name': stack_name, 'resource_status': '%s_IN_PROGRESS' % action, 'resource_status_reason': 'state changed'})
        resp_dict['events'].append({'event_time': '2013-12-05T14:14:33', 'id': stack_event2, 'links': [{'href': 'http://heat.example.com:8004/foo', 'rel': 'self'}, {'href': 'http://heat.example.com:8004/foo2', 'rel': 'resource'}, {'href': 'http://heat.example.com:8004/foo3', 'rel': 'stack'}], 'logical_resource_id': 'aResource', 'physical_resource_id': 'foo3', 'resource_name': stack_name, 'resource_status': '%s_%s' % (action, final_state), 'resource_status_reason': 'state changed'})
    return resp_dict