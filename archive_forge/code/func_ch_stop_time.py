from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError
def ch_stop_time(self, *channels: Channel) -> int:
    """Return maximum start time for supplied channels.

        Args:
            *channels: Supplied channels
        """
    if any((chan in self.channels for chan in channels)):
        return self.duration
    return 0