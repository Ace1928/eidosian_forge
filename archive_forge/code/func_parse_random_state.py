from typing import cast, Any
import numpy as np
from cirq._doc import document
def parse_random_state(random_state: RANDOM_STATE_OR_SEED_LIKE) -> np.random.RandomState:
    """Interpret an object as a pseudorandom number generator.

    If `random_state` is None, returns the module `np.random`.
    If `random_state` is an integer, returns
    `np.random.RandomState(random_state)`.
    Otherwise, returns `random_state` unmodified.

    Args:
        random_state: The object to be used as or converted to a pseudorandom
            number generator.

    Returns:
        The pseudorandom number generator object.
    """
    if random_state is None:
        return cast(np.random.RandomState, np.random)
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state)
    else:
        return cast(np.random.RandomState, random_state)