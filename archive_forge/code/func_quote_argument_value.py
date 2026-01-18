from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
def quote_argument_value(argument):
    """
    Quote an argument value.

    Example:
        'with "space"'
    is converted to:
        r'"with "space""'
    """
    return wrap_exception(quote_routeros_argument_value, argument)