from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
@property
def subcommands(self):
    if self.module.params.get('subcommands') is not None:
        return self.module.params.get('subcommands')
    elif self.path in self.current_alternatives and self.current_alternatives[self.path].get('subcommands'):
        return self.current_alternatives[self.path].get('subcommands')
    return None