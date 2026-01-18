from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterator
from qiskit import pulse, circuit
from qiskit.visualization.pulse_v2.types import PhaseFreqTuple, PulseInstruction
def get_frame_changes(self) -> Iterator[PulseInstruction]:
    """Return frame change type instructions with total frame change amount."""
    sorted_frame_changes = sorted(self._frames.items(), key=lambda x: x[0])
    phase = self._init_phase
    frequency = self._init_frequency
    for t0, frame_changes in sorted_frame_changes:
        is_opaque = False
        pre_phase = phase
        pre_frequency = frequency
        phase, frequency = ChannelEvents._calculate_current_frame(frame_changes=frame_changes, phase=phase, frequency=frequency)
        frame = PhaseFreqTuple(phase - pre_phase, frequency - pre_frequency)
        if isinstance(phase, circuit.ParameterExpression):
            phase = float(phase.bind({param: 0 for param in phase.parameters}))
            is_opaque = True
        if isinstance(frequency, circuit.ParameterExpression):
            frequency = float(frequency.bind({param: 0 for param in frequency.parameters}))
            is_opaque = True
        yield PulseInstruction(t0, self._dt, frame, frame_changes, is_opaque)