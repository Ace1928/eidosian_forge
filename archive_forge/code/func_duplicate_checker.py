from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def duplicate_checker(self, items):
    unique_items = set(items)
    if len(items) != len(unique_items):
        return [element for element in unique_items if items.count(element) > 1]
    else:
        return []