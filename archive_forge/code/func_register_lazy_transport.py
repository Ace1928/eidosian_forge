import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def register_lazy_transport(prefix, module, classname):
    if prefix not in transport_list_registry:
        register_transport_proto(prefix)
    transport_list_registry.register_lazy_transport_provider(prefix, module, classname)