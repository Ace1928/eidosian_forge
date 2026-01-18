from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
@staticmethod
def new_option(option, prefix):
    new_option_name = option[len(prefix):]
    if new_option_name == 'vserver':
        new_option_name = 'path (or svm)'
    elif new_option_name == 'volume':
        new_option_name = 'path'
    return '%sendpoint:%s' % (prefix, new_option_name)