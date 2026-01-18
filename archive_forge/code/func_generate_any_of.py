import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_any_of(self):
    """
        Means that value have to be valid by any of those definitions. It can also be valid
        by all of them.

        .. code-block:: python

            {
                'anyOf': [
                    {'type': 'number', 'minimum': 10},
                    {'type': 'number', 'maximum': 5},
                ],
            }

        Valid values for this definition are 3, 4, 5, 10, 11, ... but not 8 for example.
        """
    self._any_or_one_of_count += 1
    count = self._any_or_one_of_count
    self.l('{variable}_any_of_count{count} = 0', count=count)
    for definition_item in self._definition['anyOf']:
        with self.l('if not {variable}_any_of_count{count}:', count=count, optimize=False):
            with self.l('try:', optimize=False):
                self.generate_func_code_block(definition_item, self._variable, self._variable_name, clear_variables=True)
                self.l('{variable}_any_of_count{count} += 1', count=count)
            self.l('except JsonSchemaValueException: pass')
    with self.l('if not {variable}_any_of_count{count}:', count=count, optimize=False):
        self.exc('{name} cannot be validated by any definition', rule='anyOf')