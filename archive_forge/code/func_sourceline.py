from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
@property
def sourceline(self) -> Any:
    """XML element *sourceline* if available (lxml Element) and *elem* is set."""
    return getattr(self.elem, 'sourceline', None)