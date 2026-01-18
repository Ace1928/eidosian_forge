from collections import defaultdict, deque
from typing import Any, Counter, Dict, Iterable, Iterator, List, \
from ..exceptions import XMLSchemaValueError
from ..aliases import ModelGroupType, ModelParticleType, SchemaElementType
from ..translation import gettext as _
from .. import limits
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError
from .wildcards import XsdAnyElement, Xsd11AnyElement
from . import groups
def sort_content(content: EncodedContentType, group: ModelGroupType, default_namespace: Optional[str]=None) -> List[ContentItemType]:
    return [x for x in iter_unordered_content(content, group, default_namespace)]