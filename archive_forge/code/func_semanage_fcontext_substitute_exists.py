from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def semanage_fcontext_substitute_exists(sefcontext, target):
    """ Get the SELinux file context path substitution definition from policy. Return None if it does not exist. """
    return sefcontext.equiv_dist.get(target, sefcontext.equiv.get(target))