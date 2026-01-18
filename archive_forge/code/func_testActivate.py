import math
from collections import OrderedDict
from datetime import datetime
import pytest
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects import conversion
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
def testActivate(self):
    assert rpyp.py2rpy != robjects.conversion.get_conversion().py2rpy
    l = len(robjects.conversion.converter_ctx.get().py2rpy.registry)
    k = set(robjects.conversion.converter_ctx.get().py2rpy.registry.keys())
    rpyp.activate()
    assert len(conversion.converter_ctx.get().py2rpy.registry) > l
    rpyp.deactivate()
    assert len(conversion.converter_ctx.get().py2rpy.registry) == l
    assert set(conversion.converter_ctx.get().py2rpy.registry.keys()) == k