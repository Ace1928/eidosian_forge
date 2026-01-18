import pytest
import pennylane.numpy as pnp
import pennylane as qml
def test_load_device(self, device_kwargs):
    """Test that the device loads correctly."""
    device_kwargs['wires'] = 2
    device_kwargs['shots'] = 1234
    dev = qml.device(**device_kwargs)
    if isinstance(dev, qml.devices.Device):
        assert isinstance(dev.wires, qml.wires.Wires)
        assert dev.wires == qml.wires.Wires((0, 1))
        assert isinstance(dev.shots, qml.measurements.Shots)
        assert dev.shots == qml.measurements.Shots(1234)
        assert device_kwargs['name'] == dev.name
        assert isinstance(dev.tracker, qml.Tracker)
        return
    assert dev.num_wires == 2
    assert dev.shots == 1234
    assert dev.short_name == device_kwargs['name']