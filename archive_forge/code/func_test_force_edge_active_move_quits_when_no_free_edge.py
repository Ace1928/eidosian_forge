from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_force_edge_active_move_quits_when_no_free_edge():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    search = AnnealSequenceSearch(_create_device([q00, q01]), seed=4027383815)
    seqs, edges = search._create_initial_solution()
    assert search._force_edge_active_move((seqs, edges)) == (seqs, edges)