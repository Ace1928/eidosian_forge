from __future__ import annotations
from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping
import numpy as np
from numpy.typing import NDArray
from qiskit.result import Counts
from .shape import ShapedMixin, ShapeInput, shape_tuple
@staticmethod
def from_counts(counts: Mapping[str | int, int] | Iterable[Mapping[str | int, int]], num_bits: int | None=None) -> 'BitArray':
    """Construct a new bit array from one or more ``Counts``-like objects.

        The ``counts`` can have keys that are (uniformly) integers, hexstrings, or bitstrings.
        Their values represent numbers of occurrences of that value.

        Args:
            counts: One or more counts-like mappings with the same number of shots.
            num_bits: The desired number of bits per shot. If unset, the biggest value found sets
                this value.

        Returns:
            A new bit array with shape ``()`` for single input counts, or ``(N,)`` for an iterable
            of :math:`N` counts.

        Raises:
            ValueError: If different mappings have different numbers of shots.
            ValueError: If no counts dictionaries are supplied.
        """
    if (singleton := isinstance(counts, Mapping)):
        counts = [counts]
    else:
        counts = list(counts)
        if not counts:
            raise ValueError('At least one counts mapping expected.')
    counts = [mapping.int_outcomes() if isinstance(mapping, Counts) else mapping for mapping in counts]
    data = (v for mapping in counts for vs, count in mapping.items() for v in repeat(vs, count))
    bit_array = BitArray.from_samples(data, num_bits)
    if not singleton:
        if bit_array.num_shots % len(counts) > 0:
            raise ValueError('All of your mappings need to have the same number of shots.')
        bit_array = bit_array.reshape(len(counts), bit_array.num_shots // len(counts))
    return bit_array