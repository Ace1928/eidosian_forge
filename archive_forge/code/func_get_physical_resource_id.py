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
def get_physical_resource_id(self, stack_identifier, resource_name):
    try:
        resource = self.client.resources.get(stack_identifier, resource_name)
        return resource.physical_resource_id
    except Exception:
        raise Exception('Resource (%s) not found in stack (%s)!' % (stack_identifier, resource_name))