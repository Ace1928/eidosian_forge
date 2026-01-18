from typing import Sequence, Union, Optional, Dict, List
from collections import Counter
from copy import deepcopy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.result.result import Result
from qiskit.result.counts import Counts
from qiskit.result.distributions.probability import ProbDistribution
from qiskit.result.distributions.quasi import QuasiDistribution
from qiskit.result.postprocess import _bin_to_hex
from qiskit._accelerate import results as results_rs  # pylint: disable=no-name-in-module
def marginal_counts(result: Union[dict, Result], indices: Optional[List[int]]=None, inplace: bool=False, format_marginal: bool=False, marginalize_memory: Optional[bool]=True) -> Union[Dict[str, int], Result]:
    """Marginalize counts from an experiment over some indices of interest.

    Args:
        result: result to be marginalized
            (a Result object or a dict(str, int) of counts).
        indices: The bit positions of interest
            to marginalize over. If ``None`` (default), do not marginalize at all.
        inplace: Default: False. Operates on the original Result
            argument if True, leading to loss of original Job Result.
            It has no effect if ``result`` is a dict.
        format_marginal: Default: False. If True, takes the output of
            marginalize and formats it with placeholders between cregs and
            for non-indices.
        marginalize_memory: If True, then also marginalize the memory field (if present).
            If False, remove the memory field from the result.
            If None, leave the memory field as is.

    Returns:
        Result or dict(str, int): A Result object or a dictionary with
            the observed counts, marginalized to only account for frequency
            of observations of bits of interest.

    Raises:
        QiskitError: in case of invalid indices to marginalize over.
    """
    if isinstance(result, Result):
        if not inplace:
            result = deepcopy(result)
        for i, experiment_result in enumerate(result.results):
            counts = result.get_counts(i)
            new_counts = _marginalize(counts, indices)
            new_counts_hex = {}
            for k, v in new_counts.items():
                new_counts_hex[_bin_to_hex(k)] = v
            experiment_result.data.counts = new_counts_hex
            if indices is not None:
                experiment_result.header.memory_slots = len(indices)
                csize = getattr(experiment_result.header, 'creg_sizes', None)
                if csize is not None:
                    experiment_result.header.creg_sizes = _adjust_creg_sizes(csize, indices)
            if getattr(experiment_result.data, 'memory', None) is not None and indices is not None:
                if marginalize_memory is False:
                    delattr(experiment_result.data, 'memory')
                elif marginalize_memory is None:
                    pass
                else:
                    sorted_indices = sorted(indices, reverse=True)
                    experiment_result.data.memory = results_rs.marginal_memory(experiment_result.data.memory, sorted_indices, return_hex=True)
        return result
    else:
        marg_counts = _marginalize(result, indices)
        if format_marginal and indices is not None:
            marg_counts = _format_marginal(result, marg_counts, indices)
        return marg_counts