from typing import Generic, List, Type, TypeVar
from .errors import BzrError
from .lock import LogicalLockResult
from .pyutils import get_named_object
@classmethod
def iter_optimisers(klass):
    for provider in klass._optimisers:
        if isinstance(provider, tuple):
            yield get_named_object(provider[0], provider[1])
        else:
            yield provider