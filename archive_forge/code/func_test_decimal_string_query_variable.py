import decimal
from ..decimal import Decimal
from ..objecttype import ObjectType
from ..schema import Schema
def test_decimal_string_query_variable():
    decimal_value = decimal.Decimal('1969.1974')
    result = schema.execute('query Test($decimal: Decimal){ decimal(input: $decimal) }', variables={'decimal': decimal_value})
    assert not result.errors
    assert result.data == {'decimal': str(decimal_value)}
    assert decimal.Decimal(result.data['decimal']) == decimal_value