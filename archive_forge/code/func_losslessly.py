from collections import namedtuple
from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, cast, Any, Dict, Iterator, Iterable, \
from xml.etree.ElementTree import Element
from ..exceptions import XMLSchemaTypeError
from ..names import XSI_NAMESPACE
from ..aliases import NamespacesType, BaseXsdType
from ..namespaces import NamespaceMapper
@property
def losslessly(self) -> bool:
    """
        The XML data is decoded without loss of quality, neither on data nor on data model
        shape. Only losslessly converters can be always used to encode to an XML data that
        is strictly conformant to the schema.
        """
    return False