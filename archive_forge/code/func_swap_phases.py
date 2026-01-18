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
@versionadded(version='3.5.1', reason='The correct instruction is SWAP-PHASES, not SWAP-PHASE')
@v_args(inline=True)
def swap_phases(self, framea, frameb):
    return SWAP_PHASES(framea, frameb)