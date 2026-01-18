import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
import scipy.sparse as spa
import numpy as np
import math
def finalize_block_construction(self, pyomo_block):
    pyomo_block.inputs['Th_in'].setlb(10)
    pyomo_block.inputs['Th_in'].set_value(100)
    pyomo_block.inputs['Th_out'].setlb(10)
    pyomo_block.inputs['Th_out'].set_value(50)
    pyomo_block.inputs['Tc_in'].setlb(10)
    pyomo_block.inputs['Tc_in'].set_value(30)
    pyomo_block.inputs['Tc_out'].setlb(10)
    pyomo_block.inputs['Tc_out'].set_value(50)
    pyomo_block.inputs['UA'].set_value(100)
    pyomo_block.inputs['Q'].setlb(0)
    pyomo_block.inputs['Q'].set_value(10000)
    pyomo_block.inputs['lmtd'].setlb(0)
    pyomo_block.inputs['lmtd'].set_value(20)
    pyomo_block.inputs['dT1'].setlb(0)
    pyomo_block.inputs['dT1'].set_value(20)
    pyomo_block.inputs['dT2'].setlb(0)
    pyomo_block.inputs['dT2'].set_value(20)