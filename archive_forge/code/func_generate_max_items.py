import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_max_items(self):
    self.create_variable_is_list()
    with self.l('if {variable}_is_list:'):
        if not isinstance(self._definition['maxItems'], int):
            raise JsonSchemaDefinitionException('maxItems must be a number')
        self.create_variable_with_length()
        with self.l('if {variable}_len > {maxItems}:'):
            self.exc('{name} must contain less than or equal to {maxItems} items', rule='maxItems')