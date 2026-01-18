from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
    """Computes and returns the Jacobian with the adjoint method."""
    if self.shots is not None:
        warn('Requested adjoint differentiation to be computed with finite shots. The derivative is always exact when using the adjoint differentiation method.', UserWarning)
    tape_return_type = self._check_adjdiff_supported_measurements(tape.measurements)
    if not tape_return_type:
        return np.array([], dtype=self.state.dtype)
    if tape_return_type is State:
        raise QuantumFunctionError('This method does not support statevector return type. Use vjp method instead for this purpose.')
    self._check_adjdiff_supported_operations(tape.operations)
    processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)
    if not processed_data:
        return np.array([], dtype=self.state.dtype)
    trainable_params = processed_data['tp_shift']
    requested_threads = int(getenv('OMP_NUM_THREADS', '1'))
    adjoint_jacobian = AdjointJacobianC64() if self.use_csingle else AdjointJacobianC128()
    if self._batch_obs and requested_threads > 1:
        obs_partitions = _chunk_iterable(processed_data['obs_serialized'], requested_threads)
        jac = []
        for obs_chunk in obs_partitions:
            jac_local = adjoint_jacobian(processed_data['state_vector'], obs_chunk, processed_data['ops_serialized'], trainable_params)
            jac.extend(jac_local)
    else:
        jac = adjoint_jacobian(processed_data['state_vector'], processed_data['obs_serialized'], processed_data['ops_serialized'], trainable_params)
    jac = np.array(jac)
    jac = jac.reshape(-1, len(trainable_params))
    jac_r = np.zeros((jac.shape[0], processed_data['all_params']))
    jac_r[:, processed_data['record_tp_rows']] = jac
    if hasattr(qml, 'active_return'):
        return self._adjoint_jacobian_processing(jac_r) if qml.active_return() else jac_r
    return self._adjoint_jacobian_processing(jac_r)