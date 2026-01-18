import functools
import matplotlib as mpl
from cycler import cycler
from . import palettes
def reset_defaults():
    """Restore all RC params to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)