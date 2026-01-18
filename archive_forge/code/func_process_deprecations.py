from __future__ import absolute_import, division, print_function
from ansible.plugins import AnsiblePlugin
from ansible import constants as C
from ansible.utils.display import Display
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import (
from ansible_collections.community.hashi_vault.plugins.module_utils._connection_options import HashiVaultConnectionOptions
from ansible_collections.community.hashi_vault.plugins.module_utils._authenticator import HashiVaultAuthenticator
def process_deprecations(self, collection_name='community.hashi_vault'):
    """processes deprecations related to the collection"""
    for deprecated in list(C.config.DEPRECATED):
        name = deprecated[0]
        why = deprecated[1]['why']
        if deprecated[1].get('collection_name') != collection_name:
            continue
        if 'alternatives' in deprecated[1]:
            alt = ', use %s instead' % deprecated[1]['alternatives']
        else:
            alt = ''
        ver = deprecated[1].get('version')
        date = deprecated[1].get('date')
        collection_name = deprecated[1].get('collection_name')
        display.deprecated('%s option, %s%s' % (name, why, alt), version=ver, date=date, collection_name=collection_name)
        C.config.DEPRECATED.remove(deprecated)