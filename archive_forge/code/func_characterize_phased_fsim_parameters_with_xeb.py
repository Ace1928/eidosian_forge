import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def characterize_phased_fsim_parameters_with_xeb(sampled_df: pd.DataFrame, parameterized_circuits: List['cirq.Circuit'], cycle_depths: Sequence[int], options: XEBCharacterizationOptions, initial_simplex_step_size: float=0.1, xatol: float=0.001, fatol: float=0.001, verbose: bool=True, pool: Optional['multiprocessing.pool.Pool']=None) -> XEBCharacterizationResult:
    """Run a classical optimization to fit phased fsim parameters to experimental data, and
    thereby characterize PhasedFSim-like gates.

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
        verbose: Whether to print progress updates.
        pool: An optional multiprocessing pool to execute circuit simulations in parallel.
    """
    pair, = sampled_df['pair'].unique()
    initial_simplex, names = options.get_initial_simplex_and_names(initial_simplex_step_size=initial_simplex_step_size)
    x0 = initial_simplex[0]

    def _mean_infidelity(angles):
        params = dict(zip(names, angles))
        if verbose:
            params_str = ''
            for name, val in params.items():
                params_str += f'{name:5s} = {val:7.3g} '
            print(f'Simulating with {params_str}')
        fids = benchmark_2q_xeb_fidelities(sampled_df, parameterized_circuits, cycle_depths, param_resolver=params, pool=pool)
        loss = 1 - fids['fidelity'].mean()
        if verbose:
            print(f'Loss: {loss:7.3g}', flush=True)
        return loss
    optimization_result = optimize.minimize(_mean_infidelity, x0=x0, options={'initial_simplex': initial_simplex, 'xatol': xatol, 'fatol': fatol}, method='nelder-mead')
    final_params: 'cirq.ParamDictType' = dict(zip(names, optimization_result.x))
    fidelities_df = benchmark_2q_xeb_fidelities(sampled_df, parameterized_circuits, cycle_depths, param_resolver=final_params)
    return XEBCharacterizationResult(optimization_results={pair: optimization_result}, final_params={pair: final_params}, fidelities_df=fidelities_df)