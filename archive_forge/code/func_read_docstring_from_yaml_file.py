from __future__ import (absolute_import, division, print_function)
import ast
import tokenize
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def read_docstring_from_yaml_file(filename, verbose=True, ignore_errors=True):
    """ Read docs from 'sidecar' yaml file doc for a plugin """
    data = _init_doc_dict()
    file_data = {}
    try:
        with open(filename, 'rb') as yamlfile:
            file_data = AnsibleLoader(yamlfile.read(), file_name=filename).get_single_data()
    except Exception as e:
        msg = "Unable to parse yaml file '%s': %s" % (filename, to_native(e))
        if not ignore_errors:
            raise AnsibleParserError(msg, orig_exc=e)
        elif verbose:
            display.error(msg)
    if file_data:
        for key in string_to_vars:
            data[string_to_vars[key]] = file_data.get(key, None)
    return data