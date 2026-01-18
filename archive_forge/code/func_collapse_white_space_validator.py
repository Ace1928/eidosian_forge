import re
import math
import operator
from abc import abstractmethod
from typing import TYPE_CHECKING, cast, Any, List, Optional, Pattern, Union, \
from xml.etree.ElementTree import Element
from elementpath import XPath2Parser, XPathContext, ElementPathError, \
from ..names import XSD_LENGTH, XSD_MIN_LENGTH, XSD_MAX_LENGTH, XSD_ENUMERATION, \
from ..aliases import ElementType, SchemaType, AtomicValueType, BaseXsdType
from ..translation import gettext as _
from ..helpers import count_digits, local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaDecodeError
from .xsdbase import XsdComponent, XsdAnnotation
def collapse_white_space_validator(self, value: str) -> None:
    if '\t' in value or '\n' in value or '  ' in value:
        raise XMLSchemaValidationError(self, value, _('value contains non collapsed white spaces'))