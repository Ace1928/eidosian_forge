import functools
import itertools
import numbers
import warnings
import numpy as np
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.functions import bind_new_parameters
from pennylane.tape import QuantumScript
def process_shifts(rule, tol=1e-10, batch_duplicates=True):
    """Utility function to process gradient rules.

    Args:
        rule (array): a ``(M, N)`` array corresponding to ``M`` terms
            with parameter shifts. ``N`` has to be either ``2`` or ``3``.
            The first column corresponds to the linear combination coefficients;
            the last column contains the shift values.
            If ``N=3``, the middle column contains the multipliers.
        tol (float): floating point tolerance used when comparing shifts/coefficients
            Terms with coefficients below ``tol`` will be removed.
        batch_duplicates (bool): whether to check the input ``rule`` for duplicate
            shift values in its second column.

    Returns:
        array: The processed shift rule with small entries rounded to 0, sorted
        with respect to the absolute value of the shifts, and groups of shift
        terms with identical (multiplier and) shift fused into one term each,
        if ``batch_duplicates=True``.

    This utility function accepts coefficients and shift values as well as optionally
    multipliers, and performs the following processing:

    - Set all small (within absolute tolerance ``tol``) coefficients and shifts to 0

    - Remove terms where the coefficients are 0 (including the ones set to 0 in the previous step)

    - Terms with the same shift value (and multiplier) are combined into a single term.

    - Finally, the terms are sorted according to the absolute value of ``shift``,
      This ensures that a zero-shift term, if it exists, is returned first.
    """
    rule[np.abs(rule) < tol] = 0
    rule = rule[~(rule[:, 0] == 0)]
    if batch_duplicates:
        round_decimals = int(-np.log10(tol))
        rounded_rule = np.round(rule[:, 1:], round_decimals)
        unique_mods = np.unique(rounded_rule, axis=0)
        if rule.shape[0] != unique_mods.shape[0]:
            matches = np.all(rounded_rule[:, np.newaxis] == unique_mods[np.newaxis, :], axis=-1)
            coeffs = [np.sum(rule[slc, 0]) for slc in matches.T]
            rule = np.hstack([np.stack(coeffs)[:, np.newaxis], unique_mods])
    return rule[np.argsort(np.abs(rule[:, -1]), kind='stable')]