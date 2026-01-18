from __future__ import absolute_import, division, print_function
import abc
import os
import re
import shlex
from functools import partial
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def check_collision(t, name):
    if t in last:
        if name == last[t]:
            module.fail_json(msg='The mount point "{0}" appears twice in the {1} option'.format(t, name))
        else:
            module.fail_json(msg='The mount point "{0}" appears both in the {1} and {2} option'.format(t, name, last[t]))
    last[t] = name