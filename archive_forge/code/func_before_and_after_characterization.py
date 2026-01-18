import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def before_and_after_characterization(fidelities_df_0: pd.DataFrame, characterization_result: XEBCharacterizationResult) -> pd.DataFrame:
    """A convenience function for horizontally stacking results pre- and post- characterization
    optimization.

    Args:
        fidelities_df_0: A dataframe (before fitting), likely resulting from
            `benchmark_2q_xeb_fidelities`.
        characterization_result: The result of running a characterization. This contains the
            second fidelities dataframe as well as the new parameters.

    Returns:
          A joined dataframe with original column names suffixed by "_0" and characterized
          column names suffixed by "_c".
    """
    fit_decay_df_0 = fit_exponential_decays(fidelities_df_0)
    fit_decay_df_c = fit_exponential_decays(characterization_result.fidelities_df)
    joined_df = fit_decay_df_0.join(fit_decay_df_c, how='outer', lsuffix='_0', rsuffix='_c')
    joined_df = joined_df.reset_index().set_index('pair')
    joined_df['characterized_angles'] = [characterization_result.final_params[pair] for pair in joined_df.index]
    fp, *_ = characterization_result.final_params.values()
    for angle_name in fp.keys():
        joined_df[angle_name] = [characterization_result.final_params[pair][angle_name] for pair in joined_df.index]
    return joined_df