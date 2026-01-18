import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_provider_show(self, uuid, allocations=False):
    cmd = 'resource provider show ' + uuid
    if allocations:
        cmd = cmd + ' --allocations'
    return self.openstack(cmd, use_json=True)