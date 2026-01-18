from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterator
from qiskit import pulse, circuit
from qiskit.visualization.pulse_v2.types import PhaseFreqTuple, PulseInstruction
def get_waveforms(self) -> Iterator[PulseInstruction]:
    """Return waveform type instructions with frame."""
    sorted_frame_changes = sorted(self._frames.items(), key=lambda x: x[0], reverse=True)
    sorted_waveforms = sorted(self._waveforms.items(), key=lambda x: x[0])
    phase = self._init_phase
    frequency = self._init_frequency
    for t0, inst in sorted_waveforms:
        is_opaque = False
        while len(sorted_frame_changes) > 0 and sorted_frame_changes[-1][0] <= t0:
            _, frame_changes = sorted_frame_changes.pop()
            phase, frequency = ChannelEvents._calculate_current_frame(frame_changes=frame_changes, phase=phase, frequency=frequency)
        if isinstance(phase, circuit.ParameterExpression):
            phase = float(phase.bind({param: 0 for param in phase.parameters}))
        if isinstance(frequency, circuit.ParameterExpression):
            frequency = float(frequency.bind({param: 0 for param in frequency.parameters}))
        frame = PhaseFreqTuple(phase, frequency)
        if isinstance(inst, pulse.Play):
            is_opaque = inst.pulse.is_parameterized()
        yield PulseInstruction(t0, self._dt, frame, inst, is_opaque)