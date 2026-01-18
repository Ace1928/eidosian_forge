from . import _base
from ._axes import *
from ._axes import Axes as Subplot
def subplot_class_factory(cls):
    return cls