from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Iterator, Set, Union
from .exceptions import ElementPathTypeError
from .protocols import XsdTypeProtocol, XsdAttributeProtocol, XsdElementProtocol, \
from .datatypes import AtomicValueType
from .etree import is_etree_element
from .xpath_context import XPathSchemaContext
@abstractmethod
def iter_atomic_types(self) -> Iterator[XsdTypeProtocol]:
    """
        Returns an iterator for not builtin atomic types defined in the schema's scope. A concrete
        implementation must yield objects that implement the protocol `XsdTypeProtocol`.
        """