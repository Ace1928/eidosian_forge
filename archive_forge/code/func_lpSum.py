from collections import Counter
import sys
import warnings
from time import time
from .apis import LpSolverDefault, PULP_CBC_CMD
from .apis.core import clock
from .utilities import value
from . import constants as const
from . import mps_lp as mpslp
import logging
import re
def lpSum(vector):
    """
    Calculate the sum of a list of linear expressions

    :param vector: A list of linear expressions
    """
    return LpAffineExpression().addInPlace(vector)