from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def create_variable_keys(self):
    """
        Append code for creating variable with keys of that variable (dictionary)
        with a name ``{variable}_keys``. Similar to `create_variable_with_length`.
        """
    variable_name = '{}_keys'.format(self._variable)
    if variable_name in self._variables:
        return
    self._variables.add(variable_name)
    self.l('{variable}_keys = set({variable}.keys())')