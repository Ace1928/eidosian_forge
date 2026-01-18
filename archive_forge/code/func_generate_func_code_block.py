from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def generate_func_code_block(self, definition, variable, variable_name, clear_variables=False):
    """
        Creates validation rules for current definition.

        Returns the number of validation rules generated as code.
        """
    backup = (self._definition, self._variable, self._variable_name)
    self._definition, self._variable, self._variable_name = (definition, variable, variable_name)
    if clear_variables:
        backup_variables = self._variables
        self._variables = set()
    count = self._generate_func_code_block(definition)
    self._definition, self._variable, self._variable_name = backup
    if clear_variables:
        self._variables = backup_variables
    return count