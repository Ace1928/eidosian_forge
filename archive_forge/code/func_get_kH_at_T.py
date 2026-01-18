from ._util import get_backend
from .util.pyutil import defaultnamedtuple, deprecated
from .units import default_units
@deprecated('0.3.1', '0.5.0', __call__)
def get_kH_at_T(self, *args, **kwargs):
    return self(*args, **kwargs)