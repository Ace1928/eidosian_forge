from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.configparser import ConfigParser
from ansible.module_utils.common.text.converters import to_native
def from_ini(obj):
    """ Read the given string as INI file and return a dict """
    if not isinstance(obj, string_types):
        raise AnsibleFilterError(f'from_ini requires a str, got {type(obj)}')
    parser = IniParser()
    try:
        parser.read_file(StringIO(obj))
    except Exception as ex:
        raise AnsibleFilterError(f'from_ini failed to parse given string: {to_native(ex)}', orig_exc=ex)
    return parser.as_dict()