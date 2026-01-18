import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
def sequence_to_onehot(sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool=False) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
      sequence: An amino acid sequence.
      mapping: A dictionary mapping amino acids to integers.
      map_unknown_to_x: If True, any amino acid that is not in the mapping will be
        mapped to the unknown amino acid 'X'. If the mapping doesn't contain amino acid 'X', an error will be thrown.
        If False, any amino acid not in the mapping will throw an error.

    Returns:
      A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of the sequence.

    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_aas - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1
    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError('The mapping must have values from 0 to num_unique_aas-1 without any gaps. Got: %s' % sorted(mapping.values()))
    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)
    for aa_index, aa_type in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping['X'])
            else:
                raise ValueError(f'Invalid character in the sequence: {aa_type}')
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1
    return one_hot_arr