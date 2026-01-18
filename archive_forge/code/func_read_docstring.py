from __future__ import (absolute_import, division, print_function)
import ast
import tokenize
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def read_docstring(filename, verbose=True, ignore_errors=True):
    """ returns a documentation dictionary from Ansible plugin docstrings """
    if filename.endswith(C.YAML_DOC_EXTENSIONS):
        docstring = read_docstring_from_yaml_file(filename, verbose=verbose, ignore_errors=ignore_errors)
    elif filename.endswith(C.PYTHON_DOC_EXTENSIONS):
        docstring = read_docstring_from_python_module(filename, verbose=verbose, ignore_errors=ignore_errors)
    elif not ignore_errors:
        raise AnsibleError('Unknown documentation format: %s' % to_native(filename))
    if not docstring and (not ignore_errors):
        raise AnsibleError('Unable to parse documentation for: %s' % to_native(filename))
    docstring['seealso'] = None
    return docstring