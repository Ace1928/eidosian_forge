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
def draws_xr(self, vars: Union[str, List[str], None]=None, inc_warmup: bool=False, inc_sample: bool=False) -> 'xr.Dataset':
    """
        Returns the generated quantities draws as a xarray Dataset.

        This method can only be called when the underlying fit was made
        through sampling, it cannot be used on MLE or VB outputs.

        :param vars: optional list of variable names.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the MCMC sample, then the warmup draws are included.
            Default value is ``False``.

        See Also
        --------
        CmdStanGQ.draws
        CmdStanGQ.draws_pd
        CmdStanMCMC.draws_xr
        """
    if not XARRAY_INSTALLED:
        raise RuntimeError('Package "xarray" is not installed, cannot produce draws array.')
    if not isinstance(self.previous_fit, CmdStanMCMC):
        raise RuntimeError('Method "draws_xr" is only available when original fit is done via Sampling.')
    mcmc_vars_list = []
    dup_vars = []
    if vars is not None:
        if isinstance(vars, str):
            vars_list = [vars]
        else:
            vars_list = vars
        for var in vars_list:
            if var not in self._metadata.stan_vars:
                if inc_sample and var in self.previous_fit._metadata.stan_vars:
                    mcmc_vars_list.append(var)
                    dup_vars.append(var)
                else:
                    raise ValueError('Unknown variable: {}'.format(var))
    else:
        vars_list = list(self._metadata.stan_vars.keys())
        if inc_sample:
            for var in self.previous_fit._metadata.stan_vars.keys():
                if var not in vars_list and var not in mcmc_vars_list:
                    mcmc_vars_list.append(var)
    for var in dup_vars:
        vars_list.remove(var)
    self._assemble_generated_quantities()
    num_draws = self.previous_fit.num_draws_sampling
    sample_config = self.previous_fit._metadata.cmdstan_config
    attrs: MutableMapping[Hashable, Any] = {'stan_version': f'{sample_config['stan_version_major']}.{sample_config['stan_version_minor']}.{sample_config['stan_version_patch']}', 'model': sample_config['model'], 'num_draws_sampling': num_draws}
    if inc_warmup and sample_config['save_warmup']:
        num_draws += self.previous_fit.num_draws_warmup
        attrs['num_draws_warmup'] = self.previous_fit.num_draws_warmup
    data: MutableMapping[Hashable, Any] = {}
    coordinates: MutableMapping[Hashable, Any] = {'chain': self.chain_ids, 'draw': np.arange(num_draws)}
    for var in vars_list:
        build_xarray_data(data, self._metadata.stan_vars[var], self.draws(inc_warmup=inc_warmup))
    if inc_sample:
        for var in mcmc_vars_list:
            build_xarray_data(data, self.previous_fit._metadata.stan_vars[var], self.previous_fit.draws(inc_warmup=inc_warmup))
    return xr.Dataset(data, coords=coordinates, attrs=attrs).transpose('chain', 'draw', ...)