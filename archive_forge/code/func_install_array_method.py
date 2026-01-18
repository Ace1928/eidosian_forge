import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def install_array_method(name, generic, prefer_literal=True):
    my_attr = {'key': 'array.' + name, 'generic': generic, 'prefer_literal': prefer_literal}
    temp_class = type('Array_' + name, (AbstractTemplate,), my_attr)

    def array_attribute_attachment(self, ary):
        return types.BoundFunction(temp_class, ary)
    setattr(ArrayAttribute, 'resolve_' + name, array_attribute_attachment)