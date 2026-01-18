import decimal
from .draft04 import CodeGeneratorDraft04, JSON_TYPE_TO_PYTHON_TYPE
from .exceptions import JsonSchemaDefinitionException
from .generator import enforce_list
def generate_exclusive_minimum(self):
    with self.l('if isinstance({variable}, (int, float, Decimal)):'):
        if not isinstance(self._definition['exclusiveMinimum'], (int, float, decimal.Decimal)):
            raise JsonSchemaDefinitionException('exclusiveMinimum must be an integer, a float or a decimal')
        with self.l('if {variable} <= {exclusiveMinimum}:'):
            self.exc('{name} must be bigger than {exclusiveMinimum}', rule='exclusiveMinimum')