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
def stack_list_resp_dict(self, show_nested=False, include_project=False):
    stack1 = {'id': '1', 'stack_name': 'teststack', 'stack_owner': 'testowner', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2012-10-25T01:58:47Z'}
    stack2 = {'id': '2', 'stack_name': 'teststack2', 'stack_owner': 'testowner', 'stack_status': 'IN_PROGRESS', 'creation_time': '2012-10-25T01:58:47Z'}
    if include_project:
        stack1['project'] = 'testproject'
        stack1['project'] = 'testproject'
    resp_dict = {'stacks': [stack1, stack2]}
    if show_nested:
        nested = {'id': '3', 'stack_name': 'teststack_nested', 'stack_status': 'IN_PROGRESS', 'creation_time': '2012-10-25T01:58:47Z', 'parent': 'theparentof3'}
        if include_project:
            nested['project'] = 'testproject'
        resp_dict['stacks'].append(nested)
    return resp_dict