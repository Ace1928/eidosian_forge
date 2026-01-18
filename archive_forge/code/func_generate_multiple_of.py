import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def generate_multiple_of(self):
    with self.l('if isinstance({variable}, (int, float, Decimal)):'):
        if not isinstance(self._definition['multipleOf'], (int, float, decimal.Decimal)):
            raise JsonSchemaDefinitionException('multipleOf must be a number')
        if isinstance(self._definition['multipleOf'], float):
            self.l('quotient = Decimal(repr({variable})) / Decimal(repr({multipleOf}))')
        else:
            self.l('quotient = {variable} / {multipleOf}')
        with self.l('if int(quotient) != quotient:'):
            self.exc('{name} must be multiple of {multipleOf}', rule='multipleOf')