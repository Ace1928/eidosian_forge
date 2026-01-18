import numpy as np
from onnx.reference.op_run import OpRun
def need_context(self) -> bool:
    """The operator Loop needs to know all results produced
        so far as the loop may silently access one of them.
        Some information are not always referred in the list of inputs
        (kind of static variables).
        """
    return True