import os
import time
from cinderclient.v3 import client as cinderclient
import fixtures
from glanceclient import client as glanceclient
from keystoneauth1.exceptions import discovery as discovery_exc
from keystoneauth1 import identity
from keystoneauth1 import session as ksession
from keystoneclient import client as keystoneclient
from keystoneclient import discover as keystone_discover
from neutronclient.v2_0 import client as neutronclient
import openstack.config
import openstack.config.exceptions
from oslo_utils import uuidutils
import tempest.lib.cli.base
import testtools
import novaclient
import novaclient.api_versions
from novaclient import base
import novaclient.client
from novaclient.v2 import networks
import novaclient.v2.shell
def another_nova(self, action, flags='', params='', fail_ok=False, endpoint_type='publicURL', merge_stderr=False):
    flags += ' --os-compute-api-version %s ' % self.COMPUTE_API_VERSION
    return self.cli_clients_2.nova(action, flags, params, fail_ok, endpoint_type, merge_stderr)