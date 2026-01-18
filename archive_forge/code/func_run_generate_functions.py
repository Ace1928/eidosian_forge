from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def run_generate_functions(self, definition):
    """Returns the number of generate functions that were executed."""
    count = 0
    for key, func in self._json_keywords_to_function.items():
        if key in definition:
            func()
            count += 1
    return count