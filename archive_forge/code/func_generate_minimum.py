import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_minimum(self):
    with self.l('if isinstance({variable}, (int, float, Decimal)):'):
        if not isinstance(self._definition['minimum'], (int, float, decimal.Decimal)):
            raise JsonSchemaDefinitionException('minimum must be a number')
        if self._definition.get('exclusiveMinimum', False):
            with self.l('if {variable} <= {minimum}:'):
                self.exc('{name} must be bigger than {minimum}', rule='minimum')
        else:
            with self.l('if {variable} < {minimum}:'):
                self.exc('{name} must be bigger than or equal to {minimum}', rule='minimum')