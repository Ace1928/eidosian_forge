import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_min_properties(self):
    self.create_variable_is_dict()
    with self.l('if {variable}_is_dict:'):
        if not isinstance(self._definition['minProperties'], int):
            raise JsonSchemaDefinitionException('minProperties must be a number')
        self.create_variable_with_length()
        with self.l('if {variable}_len < {minProperties}:'):
            self.exc('{name} must contain at least {minProperties} properties', rule='minProperties')