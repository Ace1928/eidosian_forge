from __future__ import absolute_import, division, print_function
import re
import sys
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def split_nlf(self, license_code):
    """ A NLF file may contain several licenses
            One license per line
            Return a list of 1 or more licenses
        """
    licenses = license_code.count('"statusResp"')
    if licenses <= 1:
        return [license_code]
    nlfs = license_code.splitlines()
    if len(nlfs) != licenses:
        self.module.fail_json(msg='Error: unexpected format found %d entries and %d lines in %s' % (licenses, len(nlfs), license_code))
    return nlfs