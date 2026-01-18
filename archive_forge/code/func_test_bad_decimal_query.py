import decimal
from ..decimal import Decimal
from ..objecttype import ObjectType
from ..schema import Schema
def test_bad_decimal_query():
    not_a_decimal = 'Nobody expects the Spanish Inquisition!'
    result = schema.execute('{ decimal(input: "%s") }' % not_a_decimal)
    assert result.errors
    assert len(result.errors) == 1
    assert result.data is None
    assert result.errors[0].message == 'Expected value of type \'Decimal\', found "Nobody expects the Spanish Inquisition!".'
    result = schema.execute('{ decimal(input: true) }')
    assert result.errors
    assert len(result.errors) == 1
    assert result.data is None
    assert result.errors[0].message == "Expected value of type 'Decimal', found true."
    result = schema.execute('{ decimal(input: 1.2) }')
    assert result.errors
    assert len(result.errors) == 1
    assert result.data is None
    assert result.errors[0].message == "Expected value of type 'Decimal', found 1.2."