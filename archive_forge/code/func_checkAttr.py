from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
def checkAttr(self, key, a, default=None):
    return getattr(self[key], a, default) if key in self else default