from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def load_additional_registries(self):
    from . import cffi_utils, cmathdecl, enumdecl, listdecl, mathdecl, npydecl, setdecl, dictdecl
    self.install_registry(cffi_utils.registry)
    self.install_registry(cmathdecl.registry)
    self.install_registry(enumdecl.registry)
    self.install_registry(listdecl.registry)
    self.install_registry(mathdecl.registry)
    self.install_registry(npydecl.registry)
    self.install_registry(setdecl.registry)
    self.install_registry(dictdecl.registry)