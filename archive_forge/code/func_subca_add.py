from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def subca_add(self, subca_name=None, subject_dn=None, details=None):
    item = dict(ipacasubjectdn=subject_dn)
    subca_desc = details.get('description', None)
    if subca_desc is not None:
        item.update(description=subca_desc)
    return self._post_json(method='ca_add', name=subca_name, item=item)