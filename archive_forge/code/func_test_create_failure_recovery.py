import collections
import copy
import datetime
import json
import logging
import time
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from heat.common import context
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.db import api as db_api
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_create_failure_recovery(self):
    """Check that rollback still works with dynamic metadata.

        This test fails the second instance.
        """
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'OverwrittenFnGetRefIdType', 'Properties': {'Foo': 'abc'}}, 'BResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'AResource'}}}}}
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl), disable_rollback=True)

    class FakeException(Exception):
        pass
    mock_create = self.patchobject(generic_rsrc.ResourceWithFnGetRefIdType, 'handle_create', side_effect=[FakeException, None])
    mock_delete = self.patchobject(generic_rsrc.ResourceWithFnGetRefIdType, 'handle_delete', return_value=None)
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.FAILED), self.stack.state)
    self.assertEqual('abc', self.stack['AResource'].properties['Foo'])
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl), disable_rollback=True)
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual('abc', self.stack['AResource']._stored_properties_data['Foo'])
    self.assertEqual('ID-AResource', self.stack['BResource']._stored_properties_data['Foo'])
    mock_delete.assert_called_once_with()
    self.assertEqual(2, mock_create.call_count)