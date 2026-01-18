from __future__ import absolute_import, division, print_function
import sys
from ansible.module_utils.common.text.converters import to_native, to_bytes
def quote_routeros_argument(argument):

    def check_attribute(attribute):
        if ' ' in attribute:
            raise ParseError('Attribute names must not contain spaces')
        return attribute
    if '=' not in argument:
        check_attribute(argument)
        return argument
    attribute, value = argument.split('=', 1)
    check_attribute(attribute)
    value = quote_routeros_argument_value(value)
    return '%s=%s' % (attribute, value)