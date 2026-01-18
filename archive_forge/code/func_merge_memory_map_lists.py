import itertools
from typing import Dict, List, Tuple, cast
import numpy as np
from pyquil.paulis import PauliTerm
def merge_memory_map_lists(mml1: List[Dict[str, List[float]]], mml2: List[Dict[str, List[float]]]) -> List[Dict[str, List[float]]]:
    """
    Given two lists of memory maps, produce the "cartesian product" of the memory maps:

        merge_memory_map_lists([{a: 1}, {a: 2}], [{b: 3, c: 4}, {b: 5, c: 6}])

        -> [{a: 1, b: 3, c: 4}, {a: 1, b: 5, c: 6}, {a: 2, b: 3, c: 4}, {a: 2, b: 5, c: 6}]

    :param mml1: The first memory map list.
    :param mml2: The second memory map list.
    :return: A list of the merged memory maps.
    """
    if not mml1:
        return mml2
    if not mml2:
        return mml1
    return [{**d1, **d2} for d1, d2 in itertools.product(mml1, mml2)]