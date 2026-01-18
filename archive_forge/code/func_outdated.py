from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import os
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import platform
def outdated(self):
    """ Returns the list of installed, but outdated apps """
    if self._outdated is None:
        self._outdated = self.get_current_state('outdated')
    return self._outdated