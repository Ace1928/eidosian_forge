import sys
import os
import re
import pkg_resources
from string import Template
def render_template(content, variables):
    """
    Return a bytestring representing a templated file based on the
    input (content) and the variable names defined (vars).
    """
    fsenc = sys.getfilesystemencoding()

    def to_native(s, encoding='latin-1', errors='strict'):
        if isinstance(s, str):
            return s
        return str(s, encoding, errors)
    output = Template(to_native(content, fsenc)).substitute(variables)
    if isinstance(output, str):
        output = output.encode(fsenc, 'strict')
    return output