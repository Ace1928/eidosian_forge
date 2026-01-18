import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI(stability='beta')
def grid_search(values: Iterable) -> Dict[str, Iterable]:
    """Specify a grid of values to search over.

    Values specified in a grid search are guaranteed to be sampled.

    If multiple grid search variables are defined, they are combined with the
    combinatorial product. This means every possible combination of values will
    be sampled.

    Example:

        >>> from ray import tune
        >>> param_space={
        ...   "x": tune.grid_search([10, 20]),
        ...   "y": tune.grid_search(["a", "b", "c"])
        ... }

    This will create a grid of 6 samples:
    ``{"x": 10, "y": "a"}``, ``{"x": 10, "y": "b"}``, etc.

    When specifying ``num_samples`` in the
    :class:`TuneConfig <ray.tune.tune_config.TuneConfig>`, this will specify
    the number of random samples per grid search combination.

    For instance, in the example above, if ``num_samples=4``,
    a total of 24 trials will be started -
    4 trials for each of the 6 grid search combinations.

    Args:
        values: An iterable whose parameters will be used for creating a trial grid.

    """
    return {'grid_search': values}