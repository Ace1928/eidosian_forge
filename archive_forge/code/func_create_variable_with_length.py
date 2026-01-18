from collections import OrderedDict
from decimal import Decimal
import re
from .exceptions import JsonSchemaValueException, JsonSchemaDefinitionException
from .indent import indent
from .ref_resolver import RefResolver
def create_variable_with_length(self):
    """
        Append code for creating variable with length of that variable
        (for example length of list or dictionary) with name ``{variable}_len``.
        It can be called several times and always it's done only when that variable
        still does not exists.
        """
    variable_name = '{}_len'.format(self._variable)
    if variable_name in self._variables:
        return
    self._variables.add(variable_name)
    self.l('{variable}_len = len({variable})')