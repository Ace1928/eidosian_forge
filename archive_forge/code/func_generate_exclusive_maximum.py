import decimal
from .draft04 import CodeGeneratorDraft04, JSON_TYPE_TO_PYTHON_TYPE
from .exceptions import JsonSchemaDefinitionException
from .generator import enforce_list
def generate_exclusive_maximum(self):
    with self.l('if isinstance({variable}, (int, float, Decimal)):'):
        if not isinstance(self._definition['exclusiveMaximum'], (int, float, decimal.Decimal)):
            raise JsonSchemaDefinitionException('exclusiveMaximum must be an integer, a float or a decimal')
        with self.l('if {variable} >= {exclusiveMaximum}:'):
            self.exc('{name} must be smaller than {exclusiveMaximum}', rule='exclusiveMaximum')