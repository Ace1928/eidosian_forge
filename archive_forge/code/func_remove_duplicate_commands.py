from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
def remove_duplicate_commands(commands_list):
    return sorted(set(commands_list), key=lambda x: commands_list.index(x))