from .. import axes, _docstring
from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
from .polar import PolarAxes
def get_projection_names(self):
    """Return the names of all projections currently registered."""
    return sorted(self._all_projection_types)