from copy import copy as _copy
from decimal import Decimal
from elementpath.datatypes import AbstractDateTime, Duration, AbstractBinary
from typing import cast, Any, Callable, Union, Dict, List, Optional, \
from ..exceptions import XMLSchemaValueError
from ..names import XSI_NAMESPACE, XSD_ANY_SIMPLE_TYPE, XSD_SIMPLE_TYPE, \
from ..aliases import ComponentClassType, ElementType, IterDecodeType, \
from ..translation import gettext as _
from ..helpers import get_namespace, get_qname
from .exceptions import XMLSchemaValidationError
from .xsdbase import XsdComponent, XsdAnnotation, ValidationMixin
from .simple_types import XsdSimpleType
from .wildcards import XsdAnyAttribute
def iter_value_constraints(self, use_defaults: bool=True) -> Iterator[Tuple[str, str]]:
    if use_defaults:
        for k, v in self._attribute_group.items():
            if v.fixed is not None and k:
                yield (k, v.fixed)
            elif v.default is not None and k:
                yield (k, v.default)
    else:
        for k, v in self._attribute_group.items():
            if v.fixed is not None and k:
                yield (k, v.fixed)