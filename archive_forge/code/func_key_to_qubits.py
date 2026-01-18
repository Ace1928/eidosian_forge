from collections import abc, defaultdict
import datetime
from itertools import cycle
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Union, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
import google.protobuf.json_format as json_format
import cirq
from cirq_google.api import v2
@staticmethod
def key_to_qubits(target: METRIC_KEY) -> Tuple[cirq.GridQubit, ...]:
    """Returns a tuple of qubits from a metric key.

        Raises:
           ValueError: If the metric key is a tuple of strings.
        """
    if target and isinstance(target, tuple) and all((isinstance(q, cirq.GridQubit) for q in target)):
        return target
    raise ValueError(f'The metric target {target} was not a tuple of grid qubits.')