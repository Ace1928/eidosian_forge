from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def create_apache_identifier(name):
    """
    By convention if a module is loaded via name, it appears in apache2ctl -M as
    name_module.

    Some modules don't follow this convention and we use replacements for those."""
    text_workarounds = [('shib', 'mod_shib'), ('shib2', 'mod_shib'), ('evasive', 'evasive20_module')]
    re_workarounds = [('php', re.compile('^(php\\d)\\.'))]
    for a2enmod_spelling, module_name in text_workarounds:
        if a2enmod_spelling in name:
            return module_name
    for search, reexpr in re_workarounds:
        if search in name:
            try:
                rematch = reexpr.search(name)
                return rematch.group(1) + '_module'
            except AttributeError:
                pass
    return name + '_module'