import logging
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
from pyomo.contrib.trustregion.TRF import trust_region_method, _trf_config
def grad_blackbox(args, fixed):
    a, b = args[:2]
    return [cos(a - b), -cos(a - b)]