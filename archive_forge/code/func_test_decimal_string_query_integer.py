import decimal
from ..decimal import Decimal
from ..objecttype import ObjectType
from ..schema import Schema
def test_decimal_string_query_integer():
    decimal_value = 1
    result = schema.execute('{ decimal(input: %s) }' % decimal_value)
    assert not result.errors
    assert result.data == {'decimal': str(decimal_value)}
    assert decimal.Decimal(result.data['decimal']) == decimal_value