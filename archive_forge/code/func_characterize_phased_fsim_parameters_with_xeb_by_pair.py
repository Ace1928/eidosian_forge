import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def characterize_phased_fsim_parameters_with_xeb_by_pair(sampled_df: pd.DataFrame, parameterized_circuits: List['cirq.Circuit'], cycle_depths: Sequence[int], options: XEBCharacterizationOptions, initial_simplex_step_size: float=0.1, xatol: float=0.001, fatol: float=0.001, pool: Optional['multiprocessing.pool.Pool']=None) -> XEBCharacterizationResult:
    """Run a classical optimization to fit phased fsim parameters to experimental data, and
    thereby characterize PhasedFSim-like gates grouped by pairs.

    This is appropriate if you have run parallel XEB on multiple pairs of qubits.

    The optimization is done per-pair. If you have the same pair in e.g. two different
    layers the characterization optimization will lump the data together. This is in contrast
    with the benchmarking functionality, which will always index on `(layer_i, pair_i, pair)`.

    Args:
        sampled_df: The DataFrame of sampled two-qubit probability distributions returned
            from `sample_2q_xeb_circuits`.
        parameterized_circuits: The circuits corresponding to those sampled in `sampled_df`,
            but with some gates parameterized, likely by using `parameterize_circuit`.
        cycle_depths: The depths at which circuits were truncated.
        options: A set of options that controls the classical optimization loop
            for characterizing the parameterized gates.
        initial_simplex_step_size: Set the size of the initial simplex for Nelder-Mead.
        xatol: The `xatol` argument for Nelder-Mead. This is the absolute error for convergence
            in the parameters.
        fatol: The `fatol` argument for Nelder-Mead. This is the absolute error for convergence
            in the function evaluation.
        pool: An optional multiprocessing pool to execute pair optimization in parallel. Each
            optimization (and the simulations therein) runs serially.
    """
    pairs = sampled_df['pair'].unique()
    closure = _CharacterizePhasedFsimParametersWithXebClosure(parameterized_circuits=parameterized_circuits, cycle_depths=cycle_depths, options=options, initial_simplex_step_size=initial_simplex_step_size, xatol=xatol, fatol=fatol)
    subselected_dfs = [sampled_df[sampled_df['pair'] == pair] for pair in pairs]
    if pool is not None:
        results = pool.map(closure, subselected_dfs)
    else:
        results = [closure(df) for df in subselected_dfs]
    optimization_results = {}
    all_final_params = {}
    fid_dfs = []
    for result in results:
        optimization_results.update(result.optimization_results)
        all_final_params.update(result.final_params)
        fid_dfs.append(result.fidelities_df)
    return XEBCharacterizationResult(optimization_results=optimization_results, final_params=all_final_params, fidelities_df=pd.concat(fid_dfs))