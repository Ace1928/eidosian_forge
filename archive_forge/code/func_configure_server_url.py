from __future__ import absolute_import, division, print_function
import os
import sys
from ansible_collections.community.general.plugins.module_utils import redhat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import urllib, xmlrpc_client
def configure_server_url(self, server_url):
    """
            Configure server_url for registration
        """
    self.config.set('serverURL', server_url)
    self.config.save()