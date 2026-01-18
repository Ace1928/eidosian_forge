import random
import warnings
from itertools import repeat
def cxTwoPoints(ind1, ind2):
    """
    .. deprecated:: 1.0
       The function has been renamed.  Use :func:`~deap.tools.cxTwoPoint` instead.
    """
    warnings.warn('tools.cxTwoPoints has been renamed. Use cxTwoPoint instead.', FutureWarning)
    return cxTwoPoint(ind1, ind2)