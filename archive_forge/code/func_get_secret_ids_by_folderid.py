from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils import six
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
def get_secret_ids_by_folderid(self, term):
    display.debug('tss_lookup term: %s' % term)
    folder_id = self._term_to_folder_id(term)
    display.vvv(u"Secret Server lookup of Secret id's with Folder ID %d" % folder_id)
    return self._client.get_secret_ids_by_folderid(folder_id)