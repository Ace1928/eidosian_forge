from collections import namedtuple
from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, cast, Any, Dict, Iterator, Iterable, \
from xml.etree.ElementTree import Element
from ..exceptions import XMLSchemaTypeError
from ..names import XSI_NAMESPACE
from ..aliases import NamespacesType, BaseXsdType
from ..namespaces import NamespaceMapper
def map_attributes(self, attributes: Iterable[Tuple[str, Any]]) -> Iterator[Tuple[str, Any]]:
    """
        Creates an iterator for converting decoded attributes to a data structure with
        appropriate prefixes. If the instance has a not-empty map of namespaces registers
        the mapped URIs and prefixes.

        :param attributes: A sequence or an iterator of couples with the name of         the attribute and the decoded value. Default is `None` (for `simpleType`         elements, that don't have attributes).
        """
    if self.attr_prefix is None or not attributes:
        return
    else:
        for name, value in attributes:
            yield (self.attr_prefix + self.map_qname(name), value)