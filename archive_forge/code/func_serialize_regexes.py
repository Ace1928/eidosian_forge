from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def serialize_regexes(patterns_dict):
    regex_patterns = (repr(k) + ': ' + repr_regex(v) for k, v in patterns_dict.items())
    return '{\n    ' + ',\n    '.join(regex_patterns) + '\n}'