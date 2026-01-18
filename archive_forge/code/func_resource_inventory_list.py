import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_inventory_list(self, uuid, *, include_used=False):
    resources = self.openstack(f'resource provider inventory list {uuid}', use_json=True)
    if not include_used:
        for resource in resources:
            del resource['used']
    return resources