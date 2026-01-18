import inspect
from collections.abc import Mapping
from functools import partial
from .argument import Argument, to_arguments
from .mountedtype import MountedType
from .resolver import default_resolver
from .structures import NonNull
from .unmountedtype import UnmountedType
from .utils import get_type
from ..utils.deprecated import warn_deprecation
def source_resolver(source, root, info, **args):
    resolved = default_resolver(source, None, root, info, **args)
    if inspect.isfunction(resolved) or inspect.ismethod(resolved):
        return resolved()
    return resolved