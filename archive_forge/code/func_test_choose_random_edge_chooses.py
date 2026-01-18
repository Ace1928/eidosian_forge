from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_choose_random_edge_chooses():
    q00, q11, q22 = [cirq.GridQubit(x, x) for x in range(3)]
    e0, e1, e2 = ((q00, q11), (q11, q22), (q22, q00))
    search = AnnealSequenceSearch(_create_device([]), seed=4027383827)
    assert search._choose_random_edge(set()) is None
    assert search._choose_random_edge({e0}) == e0
    assert search._choose_random_edge({e0, e1, e2}) in [e0, e1, e2]