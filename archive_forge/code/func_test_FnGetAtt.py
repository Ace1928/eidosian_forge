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
def test_FnGetAtt(self):
    self.stack = self.create_stack()
    self.m_gs.return_value = ['SUCCESS']
    self.stack.create()
    rsrc = self.stack['WaitForTheHandle']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    wc_att = rsrc.FnGetAtt('Data')
    self.assertEqual(str({}), wc_att)
    handle = self.stack['WaitHandle']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), handle.state)
    test_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '123'}
    ret = handle.handle_signal(test_metadata)
    wc_att = rsrc.FnGetAtt('Data')
    self.assertEqual('{"123": "foo"}', wc_att)
    self.assertEqual('status:SUCCESS reason:bar', ret)
    test_metadata = {'Data': 'dog', 'Reason': 'cat', 'Status': 'SUCCESS', 'UniqueId': '456'}
    ret = handle.handle_signal(test_metadata)
    wc_att = rsrc.FnGetAtt('Data')
    self.assertIsInstance(wc_att, str)
    self.assertEqual({'123': 'foo', '456': 'dog'}, json.loads(wc_att))
    self.assertEqual('status:SUCCESS reason:cat', ret)
    self.assertEqual(1, self.m_gs.call_count)
    self.assertEqual(1, self.m_id.call_count)