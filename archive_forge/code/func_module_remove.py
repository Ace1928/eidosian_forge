import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def module_remove(self, instance, module):
    """Remove a module from an instance.
        """
    url = '/instances/%s/modules/%s' % (base.getid(instance), base.getid(module))
    resp, body = self.api.client.delete(url)
    common.check_for_exceptions(resp, body, url)