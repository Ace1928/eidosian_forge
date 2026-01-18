from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
from ..plugin_utils._hashi_vault_plugin import HashiVaultPlugin
def parse_kev_term(self, term, plugin_name, first_unqualified=None):
    """parses a term string into a dictionary"""
    param_dict = {}
    for i, param in enumerate(term.split()):
        try:
            key, value = param.split('=', 1)
        except ValueError:
            if i == 0 and first_unqualified is not None:
                key = first_unqualified
                value = param
            else:
                raise AnsibleError('%s lookup plugin needs key=value pairs, but received %s' % (plugin_name, term))
        if key in param_dict:
            msg = "Duplicate key '%s' in the term string '%s'." % (key, term)
            raise AnsibleOptionsError(msg)
        param_dict[key] = value
    return param_dict