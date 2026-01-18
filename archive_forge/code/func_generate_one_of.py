import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_one_of(self):
    """
        Means that value have to be valid by only one of those definitions. It can't be valid
        by two or more of them.

        .. code-block:: python

            {
                'oneOf': [
                    {'type': 'number', 'multipleOf': 3},
                    {'type': 'number', 'multipleOf': 5},
                ],
            }

        Valid values for this definition are 3, 5, 6, ... but not 15 for example.
        """
    self._any_or_one_of_count += 1
    count = self._any_or_one_of_count
    self.l('{variable}_one_of_count{count} = 0', count=count)
    for definition_item in self._definition['oneOf']:
        with self.l('if {variable}_one_of_count{count} < 2:', count=count, optimize=False):
            with self.l('try:', optimize=False):
                self.generate_func_code_block(definition_item, self._variable, self._variable_name, clear_variables=True)
                self.l('{variable}_one_of_count{count} += 1', count=count)
            self.l('except JsonSchemaValueException: pass')
    with self.l('if {variable}_one_of_count{count} != 1:', count=count):
        dynamic = '" (" + str({variable}_one_of_count{}) + " matches found)"'
        self.exc('{name} must be valid exactly by one definition', count, append_to_msg=dynamic, rule='oneOf')