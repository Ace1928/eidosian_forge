import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
@bitsize.validator
def bitsize_validator(self, attribute, value):
    if value <= 0:
        raise ValueError(f'Bitsize for self={self!r} must be a positive integer. Found {value}.')