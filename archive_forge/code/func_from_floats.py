import sys
import warnings
from ..overrides import override, strip_boolean_result
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning, require_version
@staticmethod
def from_floats(red, green, blue):
    """Return a new Color object from red/green/blue values from 0.0 to 1.0."""
    return Color(int(red * Color.MAX_VALUE), int(green * Color.MAX_VALUE), int(blue * Color.MAX_VALUE))