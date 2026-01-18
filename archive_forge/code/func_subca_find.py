from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def subca_find(self, subca_name):
    return self._post_json(method='ca_find', name=subca_name, item=None)