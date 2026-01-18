from __future__ import (absolute_import, division, print_function)
import keyword
import random
import uuid
from collections.abc import MutableMapping, MutableSequence
from json import dumps
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.parsing.splitter import parse_kv
def load_extra_vars(loader):
    if not getattr(load_extra_vars, 'extra_vars', None):
        extra_vars = {}
        for extra_vars_opt in context.CLIARGS.get('extra_vars', tuple()):
            data = None
            extra_vars_opt = to_text(extra_vars_opt, errors='surrogate_or_strict')
            if extra_vars_opt is None or not extra_vars_opt:
                continue
            if extra_vars_opt.startswith(u'@'):
                data = loader.load_from_file(extra_vars_opt[1:])
            elif extra_vars_opt[0] in [u'/', u'.']:
                raise AnsibleOptionsError("Please prepend extra_vars filename '%s' with '@'" % extra_vars_opt)
            elif extra_vars_opt[0] in [u'[', u'{']:
                data = loader.load(extra_vars_opt)
            else:
                data = parse_kv(extra_vars_opt)
            if isinstance(data, MutableMapping):
                extra_vars = combine_vars(extra_vars, data)
            else:
                raise AnsibleOptionsError("Invalid extra vars data supplied. '%s' could not be made into a dictionary" % extra_vars_opt)
        setattr(load_extra_vars, 'extra_vars', extra_vars)
    return load_extra_vars.extra_vars