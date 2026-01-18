import itertools as it
import warnings
from functools import partial
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP, StateMP, VarianceMP
from pennylane.transforms import transform
from .general_shift_rules import (
from .gradient_transform import find_and_validate_gradient_methods
from .parameter_shift import _get_operation_recipe
from .hessian_transform import _process_jacs
def expval_hessian_param_shift(tape, argnum, method_map, diagonal_shifts, off_diagonal_shifts, f0):
    """Generate the Hessian tapes that are used in the computation of the second derivative of a
    quantum tape, using analytical parameter-shift rules to do so exactly. Also define a
    post-processing function to combine the results of evaluating the tapes into the Hessian.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        argnum (array_like[bool]): Parameter indices to differentiate
            with respect to, in form of a two-dimensional boolean ``array_like`` mask.
        method_map (dict[int, string]): The differentiation method to use for each trainable
            parameter. Can be "A" or "0", where "A" is the analytical parameter shift rule
            and "0" indicates a 0 derivative (the parameter does not affect the tape's output).
        diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift values
            for the Hessian diagonal.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple length should match the number of frequencies for that parameter.
            If unspecified, equidistant shifts are used.
        off_diagonal_shifts (list[tuple[int or float]]): List containing tuples of shift
            values for the off-diagonal entries of the Hessian.
            If provided, one tuple of shifts should be given per trainable parameter
            and the tuple should match the number of frequencies for that parameter.
            The combination of shifts into bivariate shifts is performed automatically.
            If unspecified, equidistant shifts are used.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the Hessian tapes include the original input tape, the 'f0' value is used
            instead of evaluating the input tape, reducing the number of device invocations.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
        list of generated tapes, together with a post-processing
        function to be applied to the results of the evaluated tapes
        in order to obtain the Hessian matrix.
    """
    h_dim = tape.num_params
    unshifted_coeffs = {}
    add_unshifted = f0 is None
    diag_recipes, partial_offdiag_recipes = _collect_recipes(tape, argnum, method_map, diagonal_shifts, off_diagonal_shifts)
    hessian_tapes = []
    hessian_coeffs = []
    for i, j in it.combinations_with_replacement(range(h_dim), r=2):
        if not argnum[i, j]:
            hessian_coeffs.append(None)
            continue
        if i == j:
            add_unshifted, unshifted_coeffs[i, i] = _generate_diag_tapes(tape, i, diag_recipes, add_unshifted, hessian_tapes, hessian_coeffs)
        else:
            add_unshifted, unshifted_coeffs[i, j] = _generate_offdiag_tapes(tape, (i, j), partial_offdiag_recipes, add_unshifted, hessian_tapes, hessian_coeffs)
    unshifted_coeffs = {key: val for key, val in unshifted_coeffs.items() if val is not None}

    def processing_fn(results):
        num_measurements = len(tape.measurements)
        if num_measurements == 1:
            results = tuple(((r,) for r in results))
        hessians = []
        start = 1 if unshifted_coeffs and f0 is None else 0
        r0 = results[0] if start == 1 else f0
        for i, j in it.product(range(h_dim), repeat=2):
            if j < i:
                hessians.append(hessians[j * h_dim + i])
                continue
            k = i * h_dim + j - i * (i + 1) // 2
            coeffs = hessian_coeffs[k]
            if coeffs is None or len(coeffs) == 0:
                hessian = []
                for m in range(num_measurements):
                    hessian.append(qml.math.zeros_like(results[0][m]))
                hessians.append(tuple(hessian))
                continue
            res = results[start:start + len(coeffs)]
            start = start + len(coeffs)
            unshifted_coeff = unshifted_coeffs.get((i, j), None)
            hessian = []
            for m in range(num_measurements):
                measure_res = qml.math.stack([r[m] for r in res])
                coeffs = qml.math.convert_like(coeffs, measure_res)
                hess = qml.math.tensordot(measure_res, coeffs, [[0], [0]])
                if unshifted_coeff is not None:
                    hess = hess + unshifted_coeff * r0[m]
                hess = qml.math.array(hess, like=measure_res)
                hessian.append(hess)
            hessians.append(tuple(hessian))
        hessians = tuple((tuple((h[i] for h in hessians)) for i in range(num_measurements)))
        hessians = tuple((tuple((tuple((hess[i * h_dim + j] for j in range(h_dim))) for i in range(h_dim))) for hess in hessians))
        if h_dim == 1:
            hessians = tuple((hess[0][0] for hess in hessians))
        if num_measurements == 1:
            hessians = hessians[0]
        return hessians
    return (hessian_tapes, processing_fn)