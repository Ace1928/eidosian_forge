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
def setup_clients(self, conf, admin_credentials=False):
    self.manager = clients.ClientManager(conf, admin_credentials)
    self.identity_client = self.manager.identity_client
    self.keystone_client = self.manager.keystone_client
    self.orchestration_client = self.manager.orchestration_client
    self.compute_client = self.manager.compute_client
    self.object_client = self.manager.object_client
    self.client = self.orchestration_client