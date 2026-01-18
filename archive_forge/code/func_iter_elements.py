import warnings
from collections.abc import MutableMapping
from copy import copy as _copy
from typing import TYPE_CHECKING, cast, overload, Any, Iterable, Iterator, \
from xml.etree import ElementTree
from .. import limits
from ..exceptions import XMLSchemaValueError
from ..names import XSD_GROUP, XSD_SEQUENCE, XSD_ALL, XSD_CHOICE, XSD_ELEMENT, \
from ..aliases import ElementType, NamespacesType, SchemaType, IterDecodeType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, raw_xml_encode
from ..converters import ElementData
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError, \
from .xsdbase import ValidationMixin, XsdComponent, XsdType
from .particles import ParticleMixin, OccursCalculator
from .elements import XsdElement, XsdAlternative
from .wildcards import XsdAnyElement, Xsd11AnyElement
from .models import ModelVisitor, iter_unordered_content, iter_collapsed_content
def iter_elements(self) -> Iterator[SchemaElementType]:
    """
        A generator function iterating model's elements. Raises `XMLSchemaModelDepthError`
        if the overall depth of the model groups is over `limits.MAX_MODEL_DEPTH`.
        """
    if self.max_occurs == 0:
        return
    iterators: List[Iterator[ModelParticleType]] = []
    particles = iter(self)
    while True:
        for item in particles:
            if isinstance(item, XsdGroup):
                iterators.append(particles)
                particles = iter(item)
                if len(iterators) > limits.MAX_MODEL_DEPTH:
                    raise XMLSchemaModelDepthError(self)
                break
            else:
                yield item
        else:
            try:
                particles = iterators.pop()
            except IndexError:
                return