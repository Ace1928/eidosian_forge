from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
def expval_param_shift(tape, argnum=None, shifts=None, gradient_recipes=None, f0=None, broadcast=False):
    """Generate the parameter-shift tapes and postprocessing methods required
        to compute the gradient of a gate parameter with respect to an
        expectation value.

        The returned post-processing function will output tuples instead of
    stacking resaults.

        Args:
            tape (.QuantumTape): quantum tape to differentiate
            argnum (int or list[int] or None): Trainable parameter indices to differentiate
                with respect to. If not provided, the derivatives with respect to all
                trainable indices are returned. Note that the indices are with respect to
            the list of trainable parameters.
            shifts (list[tuple[int or float]]): List containing tuples of shift values.
                If provided, one tuple of shifts should be given per trainable parameter
                and the tuple should match the number of frequencies for that parameter.
                If unspecified, equidistant shifts are assumed.
            gradient_recipes (tuple(list[list[float]] or None)): List of gradient recipes
                for the parameter-shift method. One gradient recipe must be provided
                per trainable parameter.
            f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
                and the gradient recipe contains an unshifted term, this value is used,
                saving a quantum evaluation.
            broadcast (bool): Whether or not to use parameter broadcasting to create the
                a single broadcasted tape per operation instead of one tape per shift angle.

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing a
            list of generated tapes, together with a post-processing
            function to be applied to the results of the evaluated tapes
            in order to obtain the Jacobian matrix.
    """
    argnum = argnum or tape.trainable_params
    gradient_tapes = []
    gradient_data = []
    at_least_one_unshifted = False
    for idx, _ in enumerate(tape.trainable_params):
        if idx not in argnum:
            gradient_data.append((0, [], None, None, 0))
            continue
        op, op_idx, _ = tape.get_operation(idx)
        if op.name == 'Hamiltonian':
            if tape[op_idx].return_type is not qml.measurements.Expectation:
                raise ValueError(f'Can only differentiate Hamiltonian coefficients for expectations, not {tape[op_idx].return_type.value}')
            g_tapes, h_fn = qml.gradients.hamiltonian_grad(tape, idx)
            gradient_tapes.extend(g_tapes)
            gradient_data.append((1, np.array([1.0]), h_fn, None, g_tapes[0].batch_size))
            continue
        recipe = _choose_recipe(argnum, idx, gradient_recipes, shifts, tape)
        recipe, at_least_one_unshifted, unshifted_coeff = _extract_unshifted(recipe, at_least_one_unshifted, f0, gradient_tapes, tape)
        coeffs, multipliers, op_shifts = recipe.T
        g_tapes = generate_shifted_tapes(tape, idx, op_shifts, multipliers, broadcast)
        gradient_tapes.extend(g_tapes)
        batch_size = g_tapes[0].batch_size if broadcast and g_tapes else None
        gradient_data.append((len(g_tapes), coeffs, None, unshifted_coeff, batch_size))
    num_measurements = len(tape.measurements)
    single_measure = num_measurements == 1
    num_params = len(tape.trainable_params)
    tape_specs = (single_measure, num_params, num_measurements, tape.shots)

    def processing_fn(results):
        start, r0 = (1, results[0]) if at_least_one_unshifted and f0 is None else (0, f0)
        grads = []
        for data in gradient_data:
            num_tapes, *_, unshifted_coeff, batch_size = data
            if num_tapes == 0:
                if unshifted_coeff is None:
                    grads.append(None)
                    continue
                g = _evaluate_gradient(tape, [], data, r0)
                grads.append(g)
                continue
            res = results[start:start + num_tapes] if batch_size is None else results[start]
            start = start + num_tapes
            g = _evaluate_gradient(tape, res, data, r0)
            grads.append(g)
        zero_rep = _make_zero_rep(g, single_measure, tape.shots.has_partitioned_shots)
        grads = [zero_rep if g is None else g for g in grads]
        return reorder_grads(grads, tape_specs)
    processing_fn.first_result_unshifted = at_least_one_unshifted
    return (gradient_tapes, processing_fn)