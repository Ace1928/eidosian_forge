from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.validation import (
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultOptionGroupBase
def get_hvac_connection_options(self):
    """returns kwargs to be used for constructing an hvac.Client"""

    def _filter(k, v):
        return v is not None and k not in ('validate_certs', 'ca_cert')
    hvopts = self._options.get_filtered_options(_filter, *self.OPTIONS)
    hvopts['verify'] = self._conopt_verify
    retry_action = hvopts.pop('retry_action')
    if 'retries' in hvopts:
        hvopts['session'] = self._get_custom_requests_session(new_callback=self._retry_callback_generator(retry_action), **hvopts.pop('retries'))
    return hvopts