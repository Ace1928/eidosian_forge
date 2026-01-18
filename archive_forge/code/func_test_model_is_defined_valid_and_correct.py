import pytest
import pennylane.numpy as pnp
import pennylane as qml
def test_model_is_defined_valid_and_correct(self, device_kwargs):
    """Test that the capabilities dictionary defines a valid model."""
    device_kwargs['wires'] = 1
    dev = qml.device(**device_kwargs)
    if isinstance(dev, qml.devices.Device):
        pytest.skip('test is old interface specific.')
    cap = dev.capabilities()
    assert 'model' in cap
    assert cap['model'] in ['qubit', 'cv']
    if cap['model'] == 'qubit':

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.expval(qml.Z(0))
    else:

        @qml.qnode(dev)
        def circuit():
            qml.Displacement(1.0, 1.2345, wires=0)
            return qml.expval(qml.QuadX(wires=0))
    circuit()