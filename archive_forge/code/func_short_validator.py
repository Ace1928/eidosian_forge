from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def short_validator(value: int) -> None:
    if not -2 ** 15 <= value < 2 ** 15:
        raise XMLSchemaValidationError(short_validator, value, _('value must be {:s}').format('-2^15 <= x < 2^15'))