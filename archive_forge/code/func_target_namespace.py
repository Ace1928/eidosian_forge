import re
import math
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Pattern, \
from elementpath import XPath2Parser, ElementPathError, XPathToken, XPathContext, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_QNAME, XSD_UNIQUE, XSD_KEY, XSD_KEYREF, XSD_SELECTOR, XSD_FIELD
from ..translation import gettext as _
from ..helpers import get_qname, get_extended_qname
from ..aliases import ElementType, SchemaType, NamespacesType, AtomicValueType
from .exceptions import XMLSchemaNotBuiltError
from .xsdbase import XsdComponent
from .attributes import XsdAttribute
from .wildcards import XsdAnyElement
from . import elements
@property
def target_namespace(self) -> str:
    if self.token is None:
        pass
    elif self.token.symbol == ':':
        return self.token[1].namespace or self.xpath_default_namespace
    elif self.token.symbol == '@' and self.token[0].symbol == ':':
        return self.token[0][1].namespace or self.xpath_default_namespace
    return self.schema.target_namespace