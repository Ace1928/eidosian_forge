from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
def isWKlass(obj):
    if not hasattr(obj, '__propholder_parent__'):
        return
    ph = obj.__propholder_parent__
    if not isinstance(ph, weakref_ref):
        return
    return isinstance(ph(), TypedPropertyCollection)