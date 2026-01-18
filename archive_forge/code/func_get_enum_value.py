from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import shlex
def get_enum_value(self, enum_value, prev_stop):
    """Gets the input value of an enum argument in the example string.

    This is a helper function for parse_others().

    Args:
      enum_value: The namespace value of the argument in question.
      prev_stop: The index in the example string where parsing stopped.

    Returns:
     The actual input in the example string representing the argument's value.
    """
    unparsed_string = self.example_string[prev_stop:]
    if isinstance(enum_value, str):
        possible_versions = [enum_value.lower(), enum_value.upper(), enum_value.replace('-', '_'), enum_value.replace('_', '-')]
        for version in possible_versions:
            if version in unparsed_string:
                enum_value = version
    if str(enum_value) not in unparsed_string:
        unparsed_string = unparsed_string.strip()
        arg_end = unparsed_string.find(' --')
        arg_list = unparsed_string[:arg_end].split('=') if arg_end != -1 else unparsed_string.split('=')
        enum_value = ' '.join(arg_list[1:]).strip() if len(arg_list) > 2 else arg_list[-1].strip()
    return enum_value