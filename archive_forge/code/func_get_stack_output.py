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
def get_stack_output(self, stack_identifier, output_key, validate_errors=True):
    stack = self.client.stacks.get(stack_identifier)
    return self._stack_output(stack, output_key, validate_errors)