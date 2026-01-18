from collections import namedtuple
from collections.abc import Sequence
import numbers
import math
import re
import warnings
from traitlets import (
from ipywidgets import widget_serialization
from ipydatawidgets import DataUnion, NDArrayWidget, shape_constraints
import numpy as np
def new_new(cls, *args, **kwargs):
    if not kwargs and len(args) == 1 and isinstance(args, Sequence):
        return base.__new__(cls, *args[0], **kwargs)
    return base.__new__(cls, *args, **kwargs)