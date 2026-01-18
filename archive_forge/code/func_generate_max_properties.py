import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_max_properties(self):
    self.create_variable_is_dict()
    with self.l('if {variable}_is_dict:'):
        if not isinstance(self._definition['maxProperties'], int):
            raise JsonSchemaDefinitionException('maxProperties must be a number')
        self.create_variable_with_length()
        with self.l('if {variable}_len > {maxProperties}:'):
            self.exc('{name} must contain less than or equal to {maxProperties} properties', rule='maxProperties')