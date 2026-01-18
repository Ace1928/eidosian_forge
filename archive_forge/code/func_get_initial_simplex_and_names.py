import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def get_initial_simplex_and_names(self, initial_simplex_step_size: float=0.1) -> Tuple[np.ndarray, List[str]]:
    """Get an initial simplex and parameter names for the optimization implied by these options.

        The initial simplex initiates the Nelder-Mead optimization parameter. We
        use the standard simplex of `x0 + s*basis_vec` where x0 is given by the
        `xxx_default` attributes, s is `initial_simplex_step_size` and `basis_vec`
        is a one-hot encoded vector for each parameter for which the `parameterize_xxx`
        attribute is True.

        We also return a list of parameter names so the Cirq `param_resovler`
        can be accurately constructed during optimization.
        """
    x0_list = []
    names = []
    for default, symbol in self._iter_angles_for_characterization():
        if default is None:
            raise ValueError(f'{symbol.name}_default was not set.')
        x0_list.append(default)
        names.append(symbol.name)
    x0 = np.asarray(x0_list)
    n_param = len(x0)
    initial_simplex = [x0]
    for i in range(n_param):
        basis_vec = np.eye(1, n_param, i)[0]
        initial_simplex += [x0 + initial_simplex_step_size * basis_vec]
    return (np.asarray(initial_simplex), names)