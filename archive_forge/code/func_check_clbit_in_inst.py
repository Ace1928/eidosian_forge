from __future__ import annotations
import logging
import os
import subprocess
import tempfile
import shutil
import typing
from warnings import warn
from qiskit import user_config
from qiskit.utils import optionals as _optionals
from qiskit.circuit import ControlFlowOp, Measure
from . import latex as _latex
from . import text as _text
from . import matplotlib as _matplotlib
from . import _utils
from ..utils import _trim as trim_image
from ..exceptions import VisualizationError
def check_clbit_in_inst(circuit, cregbundle):
    if cregbundle is False:
        return False
    for inst in circuit.data:
        if isinstance(inst.operation, ControlFlowOp):
            for block in inst.operation.blocks:
                if check_clbit_in_inst(block, cregbundle) is False:
                    return False
        elif inst.clbits and (not isinstance(inst.operation, Measure)):
            if cregbundle is not False:
                warn('Cregbundle set to False since an instruction needs to refer to individual classical wire', RuntimeWarning, 3)
            return False
    return True