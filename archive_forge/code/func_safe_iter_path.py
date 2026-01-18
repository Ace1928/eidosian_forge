from collections import defaultdict, deque
from typing import Any, Counter, Dict, Iterable, Iterator, List, \
from ..exceptions import XMLSchemaValueError
from ..aliases import ModelGroupType, ModelParticleType, SchemaElementType
from ..translation import gettext as _
from .. import limits
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError
from .wildcards import XsdAnyElement, Xsd11AnyElement
from . import groups
def safe_iter_path() -> Iterator[SchemaElementType]:
    iterators: List[Iterator[ModelParticleType]] = []
    particles = iter(group)
    while True:
        for item in particles:
            if isinstance(item, groups.XsdGroup):
                current_path.append(item)
                iterators.append(particles)
                particles = iter(item)
                if len(iterators) > limits.MAX_MODEL_DEPTH:
                    raise XMLSchemaModelDepthError(group)
                break
            else:
                yield item
        else:
            try:
                current_path.pop()
                particles = iterators.pop()
            except IndexError:
                return