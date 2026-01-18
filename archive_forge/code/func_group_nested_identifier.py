import os
import random
import re
import subprocess
import time
import urllib
import fixtures
from heatclient import exc as heat_exceptions
from keystoneauth1 import exceptions as kc_exceptions
from oslo_log import log as logging
from oslo_utils import timeutils
from tempest import config
import testscenarios
import testtools
from heat_integrationtests.common import clients
from heat_integrationtests.common import exceptions
def group_nested_identifier(self, stack_identifier, group_name):
    rsrc = self.client.resources.get(stack_identifier, group_name)
    physical_resource_id = rsrc.physical_resource_id
    nested_stack = self.client.stacks.get(physical_resource_id, resolve_outputs=False)
    nested_identifier = '%s/%s' % (nested_stack.stack_name, nested_stack.id)
    parent_id = stack_identifier.split('/')[-1]
    self.assertEqual(parent_id, nested_stack.parent)
    return nested_identifier