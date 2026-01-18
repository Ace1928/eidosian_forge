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
def local_cmp(a, b):
    """
        compares with only values and not keys, keys should be the same for both dicts
        :param a: dict 1
        :param b: dict 2
        :return: difference of values in both dicts
        """
    return [key for key in a if a[key] != b[key]]