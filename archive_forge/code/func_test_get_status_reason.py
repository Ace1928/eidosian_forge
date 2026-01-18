import copy
import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import environment
from heat.engine import node_data
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
def test_get_status_reason(self):
    self.stack = self.create_stack()
    rsrc = self.stack['WaitHandle']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    test_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '123'}
    ret = rsrc.handle_signal(test_metadata)
    self.assertEqual(['bar'], rsrc.get_status_reason('SUCCESS'))
    self.assertEqual('status:SUCCESS reason:bar', ret)
    test_metadata = {'Data': 'dog', 'Reason': 'cat', 'Status': 'SUCCESS', 'UniqueId': '456'}
    ret = rsrc.handle_signal(test_metadata)
    self.assertEqual(['bar', 'cat'], sorted(rsrc.get_status_reason('SUCCESS')))
    self.assertEqual('status:SUCCESS reason:cat', ret)
    test_metadata = {'Data': 'boo', 'Reason': 'hoo', 'Status': 'FAILURE', 'UniqueId': '789'}
    ret = rsrc.handle_signal(test_metadata)
    self.assertEqual(['hoo'], rsrc.get_status_reason('FAILURE'))
    self.assertEqual('status:FAILURE reason:hoo', ret)