import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
def seed_argument(self, name, index, typ):
    name = self._mangle_arg_name(name)
    self.seed_type(name, typ)
    self.arg_names[index] = name