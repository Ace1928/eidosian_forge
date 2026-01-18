from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_force_edge_active_move_calls_force_edge_active():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q02 = cirq.GridQubit(0, 2)
    q12 = cirq.GridQubit(1, 2)
    device = _create_device([q00, q01, q10, q11, q02, q12])
    search = AnnealSequenceSearch(device, seed=4027383816)
    with mock.patch.object(search, '_force_edge_active') as force_edge_active:
        solution = search._create_initial_solution()
        search._force_edge_active_move(solution)
        force_edge_active.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY)
        _, args, _ = force_edge_active.mock_calls[0]
        seqs, edge, _ = args
        for seq in seqs:
            for i in range(1, len(seq)):
                assert not (seq[i - 1] == edge[0] and seq[i] == edge[1])
                assert not (seq[i - 1] == edge[1] and seq[i] == edge[0])
        assert edge[0] in chip_as_adjacency_list(device)[edge[1]]