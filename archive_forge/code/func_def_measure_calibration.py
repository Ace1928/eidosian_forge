import json
import pkgutil
import operator
from typing import List
from deprecated import deprecated
from deprecated.sphinx import versionadded
from lark import Lark, Transformer, v_args
import numpy as np
from pyquil.quilbase import (
from pyquil.quiltwaveforms import _wf_from_dict
from pyquil.quilatom import (
from pyquil.gates import (
@v_args(inline=True)
def def_measure_calibration(self, qubit, name, instructions):
    mref = FormalArgument(name) if name else None
    dmc = DefMeasureCalibration(qubit, mref, instructions)
    return dmc