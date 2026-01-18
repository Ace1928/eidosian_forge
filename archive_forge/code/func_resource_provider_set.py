import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_provider_set(self, uuid, name, parent_provider_uuid=None):
    to_exec = 'resource provider set ' + uuid + ' --name ' + name
    if parent_provider_uuid is not None:
        to_exec += ' --parent-provider ' + parent_provider_uuid
    return self.openstack(to_exec, use_json=True)