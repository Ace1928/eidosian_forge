from qiskit.pulse.channels import DriveChannel, MeasureChannel
from qiskit.pulse.configuration import LoConfig
from qiskit.exceptions import QiskitError
def get_meas_los(self, user_lo_config):
    """Set experiment level meas LO frequencies. Use default values from job level if experiment
        level values not supplied. If experiment level and job level values not supplied, raise an
        error. If configured LO frequency is the same as default, this method returns ``None``.

        Args:
            user_lo_config (LoConfig): A dictionary of LOs to format.

        Returns:
            List[float]: A list of measurement LOs.

        Raises:
            QiskitError: When LO frequencies are missing and no default is set at job level.
        """
    _m_los = None
    if self.meas_lo_freq:
        _m_los = self.meas_lo_freq.copy()
    elif self.n_qubits:
        _m_los = [None] * self.n_qubits
    if _m_los:
        for channel, lo_freq in user_lo_config.meas_los.items():
            self.default_lo_config.check_lo(channel, lo_freq)
            _m_los[channel.index] = lo_freq
        if _m_los == self.meas_lo_freq:
            return None
        if None in _m_los:
            raise QiskitError("Invalid experiment level measurement LO's. Must either pass values for all measurement channels or pass 'default_meas_los'.")
    return _m_los