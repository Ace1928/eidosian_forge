from qiskit.pulse.channels import DriveChannel, MeasureChannel
from qiskit.pulse.configuration import LoConfig
from qiskit.exceptions import QiskitError
def get_qubit_los(self, user_lo_config):
    """Set experiment level qubit LO frequencies. Use default values from job level if
        experiment level values not supplied. If experiment level and job level values not supplied,
        raise an error. If configured LO frequency is the same as default, this method returns
        ``None``.

        Args:
            user_lo_config (LoConfig): A dictionary of LOs to format.

        Returns:
            List[float]: A list of qubit LOs.

        Raises:
            QiskitError: When LO frequencies are missing and no default is set at job level.
        """
    _q_los = None
    if self.qubit_lo_freq:
        _q_los = self.qubit_lo_freq.copy()
    elif self.n_qubits:
        _q_los = [None] * self.n_qubits
    if _q_los:
        for channel, lo_freq in user_lo_config.qubit_los.items():
            self.default_lo_config.check_lo(channel, lo_freq)
            _q_los[channel.index] = lo_freq
        if _q_los == self.qubit_lo_freq:
            return None
        if None in _q_los:
            raise QiskitError("Invalid experiment level qubit LO's. Must either pass values for all drive channels or pass 'default_qubit_los'.")
    return _q_los