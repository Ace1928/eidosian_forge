import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_allocation_show(self, consumer_uuid, columns=()):
    cmd = 'resource provider allocation show ' + consumer_uuid
    cmd += ' '.join((' --column %s' % c for c in columns))
    return self.openstack(cmd, use_json=True)