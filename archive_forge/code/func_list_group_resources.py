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
def list_group_resources(self, stack_identifier, group_name, minimal=True):
    nested_identifier = self.group_nested_identifier(stack_identifier, group_name)
    if minimal:
        return self.list_resources(nested_identifier)
    return self.client.resources.list(nested_identifier)