from .. import axes, _docstring
from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
from .polar import PolarAxes
def register_projection(cls):
    projection_registry.register(cls)