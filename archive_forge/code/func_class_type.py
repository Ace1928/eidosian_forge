import enum
import numpy as np
from .abstract import Dummy, Hashable, Literal, Number, Type
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers
@property
def class_type(self):
    """
        The type of this member's class.
        """
    return self.class_type_class(self.instance_class, self.dtype)