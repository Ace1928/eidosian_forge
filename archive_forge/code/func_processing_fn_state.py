from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def processing_fn_state(tape):
    nonlocal grad_vec
    processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)
    calculate_vjp = VectorJacobianProductC64() if self.use_csingle else VectorJacobianProductC128()
    return calculate_vjp(processed_data['state_vector'], processed_data['ops_serialized'], grad_vec, processed_data['tp_shift'])