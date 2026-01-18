from collections import defaultdict, deque
from typing import Any, Counter, Dict, Iterable, Iterator, List, \
from ..exceptions import XMLSchemaValueError
from ..aliases import ModelGroupType, ModelParticleType, SchemaElementType
from ..translation import gettext as _
from .. import limits
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError
from .wildcards import XsdAnyElement, Xsd11AnyElement
from . import groups
def iter_unordered_content(self, content: EncodedContentType, default_namespace: Optional[str]=None) -> Iterator[ContentItemType]:
    return iter_unordered_content(content, self.root, default_namespace)