from decimal import Decimal
from math import isinf, isnan
from typing import Optional, Set, SupportsFloat, Union
from xml.etree.ElementTree import Element
from elementpath import datatypes
from ..exceptions import XMLSchemaValueError
from ..translation import gettext as _
from .exceptions import XMLSchemaValidationError
def python_to_float(value: SupportsFloat) -> str:
    if isnan(value):
        return 'NaN'
    if value == float('inf'):
        return 'INF'
    if value == float('-inf'):
        return '-INF'
    return str(value)