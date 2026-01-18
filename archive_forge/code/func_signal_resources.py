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
def signal_resources(self, resources):
    for r in resources:
        if 'IN_PROGRESS' in r.resource_status:
            stack_id = self.get_resource_stack_id(r)
            self.client.resources.signal(stack_id, r.resource_name)