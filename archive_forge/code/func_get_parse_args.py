from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import shlex
def get_parse_args(self, string):
    """Gets a list of arguments in a string.

    (Note: It assumes '$' is not included in the string)

    Args:
      string: The string in question.

    Returns:
      A list of the arguments from the string.
    """
    command_name = self.command_node.name.replace('_', '-')
    command_name_start = string.find(command_name)
    if command_name_start == -1:
        return
    command_name_stop = command_name_start + len(command_name) + 1
    args_list = shlex.split(string[command_name_stop:])
    return args_list