from .. import axes, _docstring
from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
from .polar import PolarAxes
def get_projection_class(self, name):
    """Get a projection class from its *name*."""
    return self._all_projection_types[name]