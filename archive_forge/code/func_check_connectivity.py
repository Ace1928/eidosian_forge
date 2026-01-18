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
def check_connectivity(self, check_ip):

    def try_connect(ip):
        try:
            urllib.request.urlopen('http://%s/' % ip)
            return True
        except IOError:
            return False
    timeout = self.conf.connectivity_timeout
    elapsed_time = 0
    while not try_connect(check_ip):
        time.sleep(10)
        elapsed_time += 10
        if elapsed_time > timeout:
            raise exceptions.TimeoutException()