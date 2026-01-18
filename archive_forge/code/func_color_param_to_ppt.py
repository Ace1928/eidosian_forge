import weakref
from functools import partial
import bokeh
import bokeh.core.properties as bp
import param as pm
from bokeh.model import DataModel
from bokeh.models import ColumnDataSource
from ..reactive import Syncable
from .document import unlocked
from .notebook import push
from .state import state
def color_param_to_ppt(p, kwargs):
    ppt = bp.Color(**kwargs)
    ppt._help = None
    return ppt