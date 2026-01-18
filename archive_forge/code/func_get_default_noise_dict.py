from typing import List, Dict, Sequence, Any
import cirq
import cirq_pasqal
def get_default_noise_dict(self) -> Dict[str, Any]:
    """Returns the current noise parameters"""
    default_noise_dict = {str(cirq.YPowGate()): cirq.depolarize(0.01), str(cirq.ZPowGate()): cirq.depolarize(0.01), str(cirq.XPowGate()): cirq.depolarize(0.01), str(cirq.PhasedXPowGate(phase_exponent=0)): cirq.depolarize(0.01), str(cirq.HPowGate(exponent=1)): cirq.depolarize(0.01), str(cirq.CNotPowGate(exponent=1)): cirq.depolarize(0.03), str(cirq.CZPowGate(exponent=1)): cirq.depolarize(0.03), str(cirq.CCXPowGate(exponent=1)): cirq.depolarize(0.08), str(cirq.CCZPowGate(exponent=1)): cirq.depolarize(0.08)}
    return default_noise_dict