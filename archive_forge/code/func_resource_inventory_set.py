import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_inventory_set(self, uuid, *resources, **kwargs):
    opts = []
    if kwargs.get('aggregate'):
        opts.append('--aggregate')
    if kwargs.get('amend'):
        opts.append('--amend')
    if kwargs.get('dry_run'):
        opts.append('--dry-run')
    fmt = 'resource provider inventory set {uuid} {resources} {opts}'
    cmd = fmt.format(uuid=uuid, resources=' '.join(['--resource %s' % r for r in resources]), opts=' '.join(opts))
    return self.openstack(cmd, use_json=True)