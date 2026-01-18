import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_dependencies(self):
    """
        Means when object has property, it needs to have also other property.

        .. code-block:: python

            {
                'dependencies': {
                    'bar': ['foo'],
                },
            }

        Valid object is containing only foo, both bar and foo or none of them, but not
        object with only bar.

        Since draft 06 definition can be boolean or empty array. True and empty array
        means nothing, False means that key cannot be there at all.
        """
    self.create_variable_is_dict()
    with self.l('if {variable}_is_dict:'):
        is_empty = True
        for key, values in self._definition['dependencies'].items():
            if values == [] or values is True:
                continue
            is_empty = False
            with self.l('if "{}" in {variable}:', self.e(key)):
                if values is False:
                    self.exc('{} in {name} must not be there', key, rule='dependencies')
                elif isinstance(values, list):
                    for value in values:
                        with self.l('if "{}" not in {variable}:', self.e(value)):
                            self.exc('{name} missing dependency {} for {}', self.e(value), self.e(key), rule='dependencies')
                else:
                    self.generate_func_code_block(values, self._variable, self._variable_name, clear_variables=True)
        if is_empty:
            self.l('pass')