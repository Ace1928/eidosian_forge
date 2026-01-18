from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Union, Optional
from qiskit import pulse
from qiskit.providers import BackendConfigurationError
from qiskit.providers.backend import Backend
def get_channel_frequency(self, chan: pulse.channels.Channel) -> Union[float, None]:
    """Get frequency of given channel object."""
    return self._chan_freq_map.get(chan, None)