from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
def fillDerivedValues(self, node):
    """Examine a node for any values which are Derived,
        and replace them with their calculated values.
        Generally things may look at the drawing or their
        parent.

        """
    for key, value in node.__dict__.items():
        if isinstance(value, DerivedValue):
            newValue = value.getValue(self, key)
            node.__dict__[key] = newValue