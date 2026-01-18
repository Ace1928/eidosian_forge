from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Union, Optional
from qiskit import pulse
from qiskit.providers import BackendConfigurationError
from qiskit.providers.backend import Backend
@classmethod
def create_from_backend(cls, backend: Backend):
    """Initialize a class with backend information provided by provider.

        Args:
            backend: Backend object.

        Returns:
            OpenPulseBackendInfo: New configured instance.
        """
    configuration = backend.configuration()
    defaults = backend.defaults()
    name = backend.name()
    dt = configuration.dt
    chan_freqs = {}
    chan_freqs.update({pulse.DriveChannel(qind): freq for qind, freq in enumerate(defaults.qubit_freq_est)})
    chan_freqs.update({pulse.MeasureChannel(qind): freq for qind, freq in enumerate(defaults.meas_freq_est)})
    for qind, u_lo_mappers in enumerate(configuration.u_channel_lo):
        temp_val = 0.0 + 0j
        for u_lo_mapper in u_lo_mappers:
            temp_val += defaults.qubit_freq_est[u_lo_mapper.q] * u_lo_mapper.scale
        chan_freqs[pulse.ControlChannel(qind)] = temp_val.real
    qubit_channel_map = defaultdict(list)
    for qind in range(configuration.n_qubits):
        qubit_channel_map[qind].append(configuration.drive(qubit=qind))
        qubit_channel_map[qind].append(configuration.measure(qubit=qind))
        for tind in range(configuration.n_qubits):
            try:
                qubit_channel_map[qind].extend(configuration.control(qubits=(qind, tind)))
            except BackendConfigurationError:
                pass
    return OpenPulseBackendInfo(name=name, dt=dt, channel_frequency_map=chan_freqs, qubit_channel_map=qubit_channel_map)