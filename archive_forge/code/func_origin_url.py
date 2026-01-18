from typing import TYPE_CHECKING, Any, Optional, cast, Iterable, Union, Callable
from elementpath.etree import etree_tostring
from ..exceptions import XMLSchemaException, XMLSchemaWarning, XMLSchemaValueError
from ..aliases import ElementType, NamespacesType, SchemaElementType, ModelParticleType
from ..helpers import get_prefixed_qname, etree_getpath, is_etree_element
from ..translation import gettext as _
@property
def origin_url(self) -> Optional[str]:
    """The origin schema URL, if available and the *validator* is an XSD component."""
    url: Optional[str]
    try:
        url = self.validator.maps.validator.source.url
    except AttributeError:
        return None
    else:
        return url