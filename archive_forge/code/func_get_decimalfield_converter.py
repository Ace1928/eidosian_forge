import datetime
import decimal
import uuid
from functools import lru_cache
from itertools import chain
from django.conf import settings
from django.core.exceptions import FieldError
from django.db import DatabaseError, NotSupportedError, models
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.constants import OnConflict
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.functional import cached_property
def get_decimalfield_converter(self, expression):
    create_decimal = decimal.Context(prec=15).create_decimal_from_float
    if isinstance(expression, Col):
        quantize_value = decimal.Decimal(1).scaleb(-expression.output_field.decimal_places)

        def converter(value, expression, connection):
            if value is not None:
                return create_decimal(value).quantize(quantize_value, context=expression.output_field.context)
    else:

        def converter(value, expression, connection):
            if value is not None:
                return create_decimal(value)
    return converter