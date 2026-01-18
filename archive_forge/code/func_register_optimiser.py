from typing import Generic, List, Type, TypeVar
from .errors import BzrError
from .lock import LogicalLockResult
from .pyutils import get_named_object
@classmethod
def register_optimiser(klass, optimiser):
    """Register an InterObject optimiser."""
    klass._optimisers.append(optimiser)