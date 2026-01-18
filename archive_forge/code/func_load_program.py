from __future__ import annotations
from collections.abc import Iterator, Sequence
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import chain
import numpy as np
from qiskit import pulse
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import events, types, drawings, device_info
from qiskit.visualization.pulse_v2.stylesheet import QiskitPulseStyle
def load_program(self, program: pulse.Schedule, chan: pulse.channels.Channel):
    """Load pulse schedule.

        This method internally generates `ChannelEvents` to parse the program
        for the specified pulse channel. This method is called once

        Args:
            program: Pulse schedule to load.
            chan: A pulse channels associated with this instance.
        """
    chan_events = events.ChannelEvents.load_program(program, chan)
    chan_events.set_config(dt=self.parent.device.dt, init_frequency=self.parent.device.get_channel_frequency(chan), init_phase=0)
    for gen in self.parent.generator['waveform']:
        waveforms = chan_events.get_waveforms()
        obj_generator = partial(gen, formatter=self.parent.formatter, device=self.parent.device)
        drawing_items = [obj_generator(waveform) for waveform in waveforms]
        for drawing_item in list(chain.from_iterable(drawing_items)):
            self.add_data(drawing_item)
    for gen in self.parent.generator['frame']:
        frames = chan_events.get_frame_changes()
        obj_generator = partial(gen, formatter=self.parent.formatter, device=self.parent.device)
        drawing_items = [obj_generator(frame) for frame in frames]
        for drawing_item in list(chain.from_iterable(drawing_items)):
            self.add_data(drawing_item)
    self._channels.add(chan)