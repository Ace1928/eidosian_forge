from typing import cast, Any, Callable, Dict, Iterable, Iterator, List, Optional, \
from elementpath import SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaValueError
from ..names import XSI_NAMESPACE, XSD_ANY, XSD_ANY_ATTRIBUTE, \
from ..aliases import ElementType, SchemaType, SchemaElementType, SchemaAttributeType, \
from ..translation import gettext as _
from ..helpers import get_namespace, raw_xml_encode
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, ElementPathMixin
from .xsdbase import ValidationMixin, XsdComponent
from .particles import ParticleMixin
from . import elements
def add_precedence(self, other: ModelParticleType, group: ModelGroupType) -> None:
    try:
        self.precedences[group].append(other)
    except KeyError:
        self.precedences[group] = [other]