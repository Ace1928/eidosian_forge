import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_inventory_class_set(self, uuid, resource_class, **kwargs):
    opts = ['--%s=%s' % (k, v) for k, v in kwargs.items()]
    cmd = 'resource provider inventory class set {uuid} {rc} {opts}'.format(uuid=uuid, rc=resource_class, opts=' '.join(opts))
    return self.openstack(cmd, use_json=True)