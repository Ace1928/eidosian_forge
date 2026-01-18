from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.stanfit.metadata import InferenceMetadata
from cmdstanpy.stanfit.runset import RunSet
from cmdstanpy.utils.stancsv import scan_generic_csv
def create_inits(self, seed: Optional[int]=None, chains: int=4) -> Union[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
        Create initial values for the parameters of the model
        by randomly selecting draws from the Pathfinder approximation.

        :param seed: Used for random selection, defaults to None
        :param chains: Number of initial values to return, defaults to 4
        :return: The initial values for the parameters of the model.

        If ``chains`` is 1, a dictionary is returned, otherwise a list
        of dictionaries is returned, in the format expected for the
        ``inits`` argument. of :meth:`CmdStanModel.sample`.
        """
    self._assemble_draws()
    rng = np.random.default_rng(seed)
    idxs = rng.choice(self._draws.shape[0], size=chains, replace=False)
    if chains == 1:
        draw = self._draws[idxs[0]]
        return {name: var.extract_reshape(draw) for name, var in self._metadata.stan_vars.items()}
    else:
        return [{name: var.extract_reshape(self._draws[idx]) for name, var in self._metadata.stan_vars.items()} for idx in idxs]