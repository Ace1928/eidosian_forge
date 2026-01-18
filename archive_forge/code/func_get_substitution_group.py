from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Iterator, Set, Union
from .exceptions import ElementPathTypeError
from .protocols import XsdTypeProtocol, XsdAttributeProtocol, XsdElementProtocol, \
from .datatypes import AtomicValueType
from .etree import is_etree_element
from .xpath_context import XPathSchemaContext
def get_substitution_group(self, qname: str) -> Optional[Set[XsdElementProtocol]]:
    """
        Get a substitution group. A concrete implementation must returns a list containing
        substitution elements or `None` if the substitution group is not found. Moreover each item
        of the returned list must be an object that implements the `AbstractXsdElement` interface.

        :param qname: the fully qualified name of the substitution group to retrieve.
        :returns: a list containing substitution elements or `None`.
        """
    return self._schema.maps.substitution_groups.get(qname)