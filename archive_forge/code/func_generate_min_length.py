import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_min_length(self):
    with self.l('if isinstance({variable}, str):'):
        self.create_variable_with_length()
        if not isinstance(self._definition['minLength'], int):
            raise JsonSchemaDefinitionException('minLength must be a number')
        with self.l('if {variable}_len < {minLength}:'):
            self.exc('{name} must be longer than or equal to {minLength} characters', rule='minLength')