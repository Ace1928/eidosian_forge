from typing import Generic, List, Type, TypeVar
from .errors import BzrError
from .lock import LogicalLockResult
from .pyutils import get_named_object
@classmethod
def register_lazy_optimiser(klass, module_name, member_name):
    klass._optimisers.append((module_name, member_name))