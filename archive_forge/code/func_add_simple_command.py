from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.dns.plugins.module_utils.http import (
def add_simple_command(self, command, **args):
    self._command = command
    command = self._create(command, self._api)
    for arg, value in args.items():
        arg = self._create(arg)
        encode_wsdl(arg, value)
        command.append(arg)
    self._body.append(command)