import collections
import datetime
import itertools
import json
import os
import sys
from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import attributes
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import clients
from heat.engine import constraints
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import node_data
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources.openstack.heat import none_resource
from heat.engine.resources.openstack.heat import test_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.engine import translation
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_object
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
import neutronclient.common.exceptions as neutron_exp
def test_deprecated_prop_data_updated(self):
    tmpl = rsrc_defn.ResourceDefinition('test_resource', 'Foo', {'Foo': 'abc'})
    res = generic_rsrc.ResourceWithProps('test_resource', tmpl, self.stack)
    scheduler.TaskRunner(res.create)()
    res_obj = db_api.resource_get(self.stack.context, res.id)
    self.assertIsNone(res_obj.properties_data)
    self.assertIsNone(res_obj.properties_data_encrypted)
    res_obj = db_api.resource_update_and_save(self.stack.context, res_obj.id, {'properties_data': {'Foo': 'lucky'}, 'properties_data_encrypted': False, 'rsrc_prop_data': None})
    res._rsrc_prop_data = None
    res._load_data(res_obj)
    self.assertEqual(res._stored_properties_data, {'Foo': 'lucky'})
    res._rsrc_prop_data = None
    res.state_set(res.CREATE, res.IN_PROGRESS, 'test_rpd')
    rsrc_prop_data_db_obj = db_api.resource_prop_data_get(self.stack.context, res._rsrc_prop_data_id)
    self.assertEqual(rsrc_prop_data_db_obj['data'], {'Foo': 'lucky'})
    self.assertFalse(hasattr(res, 'properties_data'))
    self.assertFalse(hasattr(res, 'properties_data_encrypted'))