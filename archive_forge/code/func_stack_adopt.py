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
def stack_adopt(self, stack_name=None, files=None, parameters=None, environment=None, adopt_data=None, wait_for_status='ADOPT_COMPLETE'):
    if self.conf.skip_test_stack_action_list and 'ADOPT' in self.conf.skip_test_stack_action_list:
        self.skipTest('Testing Stack adopt disabled in conf, skipping')
    name = stack_name or self._stack_rand_name()
    templ_files = files or {}
    params = parameters or {}
    env = environment or {}
    self.client.stacks.create(stack_name=name, files=templ_files, disable_rollback=True, parameters=params, environment=env, adopt_stack_data=adopt_data)
    self.addCleanup(self._stack_delete, name)
    stack = self.client.stacks.get(name, resolve_outputs=False)
    stack_identifier = '%s/%s' % (name, stack.id)
    self._wait_for_stack_status(stack_identifier, wait_for_status)
    return stack_identifier