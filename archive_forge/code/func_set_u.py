from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .common import TestCase
import numpy as np
def set_u(q, u):
    q.uncertainty = u