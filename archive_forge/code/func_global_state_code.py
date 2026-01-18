from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
@property
def global_state_code(self):
    """
        Returns global variables for generating function from ``func_code`` as code.
        Includes compiled regular expressions and imports.
        """
    self._generate_func_code()
    if not self._compile_regexps:
        return '\n'.join(self._extra_imports_lines + ['from fastjsonschema import JsonSchemaValueException', '', ''])
    return '\n'.join(self._extra_imports_lines + ['import re', 'from fastjsonschema import JsonSchemaValueException', '', '', 'REGEX_PATTERNS = ' + serialize_regexes(self._compile_regexps), ''])