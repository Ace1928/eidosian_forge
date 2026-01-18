from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def qname_validator(value: str) -> None:
    if datatypes.QName.pattern.match(value) is None:
        raise XMLSchemaValidationError(qname_validator, value, _('value is not an xs:QName'))