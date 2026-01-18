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
def test_make_replacement(self):
    tmpl = rsrc_defn.ResourceDefinition('test_resource', 'Foo')
    res = generic_rsrc.GenericResource('test_res_upd', tmpl, self.stack)
    res.store()
    new_tmpl_id = 2
    self.assertIsNotNone(res.id)
    new_requires = {1, 2, 4}
    new_id = res.make_replacement(new_tmpl_id, new_requires)
    new_res = resource_objects.Resource.get_obj(res.context, new_id)
    self.assertEqual(new_id, res.replaced_by)
    self.assertEqual(res.id, new_res.replaces)
    self.assertIsNone(new_res.physical_resource_id)
    self.assertEqual(new_tmpl_id, new_res.current_template_id)
    self.assertEqual([4, 2, 1], new_res.requires)