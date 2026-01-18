from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_product_spec(self, switch_version):
    """Create product info spec"""
    product_info_spec = vim.dvs.ProductSpec()
    product_info_spec.version = switch_version
    return product_info_spec