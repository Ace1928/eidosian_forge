import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pyquil.experiment._setting import ExperimentSetting
def bitstrings_to_expectations(bitstrings: np.ndarray, joint_expectations: Optional[List[List[int]]]=None) -> np.ndarray:
    """
    Given an array of bitstrings (each of which is represented as an array of bits), map them to
    expectation values and return the desired joint expectation values. If no joint expectations
    are desired, then just the 1 -> -1, 0 -> 1 mapping is performed.

    :param bitstrings: Array of bitstrings to map.
    :param joint_expectations: Joint expectation values to calculate. Each entry is a list which
        contains the qubits to use in calculating the joint expectation value. Entries of length
        one just calculate single-qubit expectation values. Defaults to None, which is equivalent
        to the list of single-qubit expectations [[0], [1], ..., [n-1]] for bitstrings of length n.
    :return: An array of expectation values, of the same length as the array of bitstrings. The
        "width" could be different than the length of an individual bitstring (n) depending on
        the value of the ``joint_expectations`` parameter.
    """
    expectations: np.ndarray = 1 - 2 * bitstrings
    if joint_expectations is None:
        return expectations
    region_size = len(expectations[0])
    e = []
    for c in joint_expectations:
        where = np.zeros(region_size, dtype=bool)
        where[c] = True
        e.append(np.prod(expectations[:, where], axis=1))
    return np.stack(e, axis=-1)