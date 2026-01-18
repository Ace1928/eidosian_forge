import json
import urllib
import numpy as np
import pytest
import cirq
from cirq import quirk_url_to_circuit, quirk_json_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_extra_cell_makers():
    assert cirq.quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[["iswap"]]}', extra_cell_makers=[cirq.interop.quirk.cells.CellMaker(identifier='iswap', size=2, maker=lambda args: cirq.ISWAP(*args.qubits))]) == cirq.Circuit(cirq.ISWAP(*cirq.LineQubit.range(2)))
    assert cirq.quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[["iswap"]]}', extra_cell_makers={'iswap': cirq.ISWAP}) == cirq.Circuit(cirq.ISWAP(*cirq.LineQubit.range(2)))
    assert cirq.quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[["iswap"], ["toffoli"]]}', extra_cell_makers=[cirq.interop.quirk.cells.CellMaker(identifier='iswap', size=2, maker=lambda args: cirq.ISWAP(*args.qubits)), cirq.interop.quirk.cells.CellMaker(identifier='toffoli', size=3, maker=lambda args: cirq.TOFFOLI(*args.qubits))]) == cirq.Circuit([cirq.ISWAP(*cirq.LineQubit.range(2)), cirq.TOFFOLI(*cirq.LineQubit.range(3))])
    assert cirq.quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[["iswap"], ["toffoli"]]}', extra_cell_makers={'iswap': cirq.ISWAP, 'toffoli': cirq.TOFFOLI}) == cirq.Circuit([cirq.ISWAP(*cirq.LineQubit.range(2)), cirq.TOFFOLI(*cirq.LineQubit.range(3))])