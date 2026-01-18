from collections import Counter
from typing import (
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import build_xarray_data, flatten_chains, get_logger
from cmdstanpy.utils.stancsv import scan_generic_csv
from .mcmc import CmdStanMCMC
from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet
from .vb import CmdStanVB
def draws_pd(self, vars: Union[List[str], str, None]=None, inc_warmup: bool=False, inc_sample: bool=False) -> pd.DataFrame:
    """
        Returns the generated quantities draws as a pandas DataFrame.
        Flattens all chains into single column.  Container variables
        (array, vector, matrix) will span multiple columns, one column
        per element. E.g. variable 'matrix[2,2] foo' spans 4 columns:
        'foo[1,1], ... foo[2,2]'.

        :param vars: optional list of variable names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        See Also
        --------
        CmdStanGQ.draws
        CmdStanGQ.draws_xr
        CmdStanMCMC.draws_pd
        """
    if vars is not None:
        if isinstance(vars, str):
            vars_list = [vars]
        else:
            vars_list = vars
        vars_list = list(dict.fromkeys(vars_list))
    if inc_warmup:
        if isinstance(self.previous_fit, CmdStanMCMC) and (not self.previous_fit._save_warmup):
            get_logger().warning('Sample doesn\'t contain draws from warmup iterations, rerun sampler with "save_warmup=True".')
        elif isinstance(self.previous_fit, CmdStanMLE) and (not self.previous_fit._save_iterations):
            get_logger().warning('MLE doesn\'t contain draws from pre-convergence iterations, rerun optimization with "save_iterations=True".')
        elif isinstance(self.previous_fit, CmdStanVB):
            get_logger().warning('Variational fit doesn\'t make sense with argument "inc_warmup=True"')
    self._assemble_generated_quantities()
    all_columns = ['chain__', 'iter__', 'draw__'] + list(self.column_names)
    gq_cols: List[str] = []
    mcmc_vars: List[str] = []
    if vars is not None:
        for var in vars_list:
            if var in self._metadata.stan_vars:
                info = self._metadata.stan_vars[var]
                gq_cols.extend(self.column_names[info.start_idx:info.end_idx])
            elif inc_sample and var in self.previous_fit._metadata.stan_vars:
                info = self.previous_fit._metadata.stan_vars[var]
                mcmc_vars.extend(self.previous_fit.column_names[info.start_idx:info.end_idx])
            elif var in ['chain__', 'iter__', 'draw__']:
                gq_cols.append(var)
            else:
                raise ValueError('Unknown variable: {}'.format(var))
    else:
        gq_cols = all_columns
        vars_list = gq_cols
    previous_draws_pd = self._previous_draws_pd(mcmc_vars, inc_warmup)
    draws = self.draws(inc_warmup=inc_warmup)
    n_draws, n_chains, _ = draws.shape
    chains_col = np.repeat(np.arange(1, n_chains + 1), n_draws).reshape(1, n_chains, n_draws).T
    iter_col = np.tile(np.arange(1, n_draws + 1), n_chains).reshape(1, n_chains, n_draws).T
    draw_col = np.arange(1, n_draws * n_chains + 1).reshape(1, n_chains, n_draws).T
    draws = np.concatenate([chains_col, iter_col, draw_col, draws], axis=2)
    draws_pd = pd.DataFrame(data=flatten_chains(draws), columns=all_columns)
    if inc_sample and mcmc_vars:
        if gq_cols:
            return pd.concat([previous_draws_pd, draws_pd[gq_cols]], axis='columns')[vars_list]
        else:
            return previous_draws_pd
    elif inc_sample and vars is None:
        cols_1 = list(previous_draws_pd.columns)
        cols_2 = list(draws_pd.columns)
        dups = [item for item, count in Counter(cols_1 + cols_2).items() if count > 1]
        return pd.concat([previous_draws_pd.drop(columns=dups).reset_index(drop=True), draws_pd], axis=1)
    elif gq_cols:
        return draws_pd[gq_cols]
    return draws_pd