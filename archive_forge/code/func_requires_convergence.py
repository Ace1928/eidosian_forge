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
def requires_convergence(test_method):
    """Decorator for convergence-only tests.

    The decorated test will be skipped when convergence is disabled.
    """
    convergence_enabled = config.CONF.heat_plugin.convergence_engine_enabled
    skipper = testtools.skipUnless(convergence_enabled, 'Convergence-only tests are disabled')
    return skipper(test_method)