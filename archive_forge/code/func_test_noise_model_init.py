import pytest
import cirq
from cirq_pasqal import PasqalNoiseModel, PasqalDevice
from cirq.ops import NamedQubit
def test_noise_model_init():
    noise_model = PasqalNoiseModel(PasqalDevice([]))
    assert noise_model.noise_op_dict == {str(cirq.ops.YPowGate()): cirq.ops.depolarize(0.01), str(cirq.ops.ZPowGate()): cirq.ops.depolarize(0.01), str(cirq.ops.XPowGate()): cirq.ops.depolarize(0.01), str(cirq.ops.HPowGate(exponent=1)): cirq.ops.depolarize(0.01), str(cirq.ops.PhasedXPowGate(phase_exponent=0)): cirq.ops.depolarize(0.01), str(cirq.ops.CNotPowGate(exponent=1)): cirq.ops.depolarize(0.03), str(cirq.ops.CZPowGate(exponent=1)): cirq.ops.depolarize(0.03), str(cirq.ops.CCXPowGate(exponent=1)): cirq.ops.depolarize(0.08), str(cirq.ops.CCZPowGate(exponent=1)): cirq.ops.depolarize(0.08)}
    with pytest.raises(TypeError, match="noise model varies between Pasqal's devices."):
        PasqalNoiseModel(cirq.devices.UNCONSTRAINED_DEVICE)