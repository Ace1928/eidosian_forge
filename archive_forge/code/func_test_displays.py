import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_displays():
    assert_url_to_circuit_returns('{"cols":[["Amps2"],[1,"Amps3"],["Chance"],["Chance2"],["Density"],["Density3"],["Sample4"],["Bloch"],["Sample2"]]}', cirq.Circuit())