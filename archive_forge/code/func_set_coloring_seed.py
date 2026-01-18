import itertools
import functools
import importlib.util
def set_coloring_seed(seed):
    """Set the seed for the random color generator.

    Parameters
    ----------
    seed : int
        The seed to use.
    """
    global COLORING_SEED
    COLORING_SEED = seed