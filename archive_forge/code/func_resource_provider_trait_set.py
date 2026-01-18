import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_provider_trait_set(self, uuid, *traits):
    cmd = 'resource provider trait set %s ' % uuid
    cmd += ' '.join(('--trait %s' % trait for trait in traits))
    return self.openstack(cmd, use_json=True)