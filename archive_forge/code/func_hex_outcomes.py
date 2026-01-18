import re
from qiskit.result import postprocess
from qiskit import exceptions
def hex_outcomes(self):
    """Return a counts dictionary with hexadecimal string keys

        Returns:
            dict: A dictionary with the keys as hexadecimal strings instead of
                bitstrings
        Raises:
            QiskitError: If the Counts object contains counts for dit strings
        """
    if self.hex_raw:
        return {key.lower(): value for key, value in self.hex_raw.items()}
    else:
        out_dict = {}
        for bitstring, value in self.items():
            if not self.bitstring_regex.search(bitstring):
                raise exceptions.QiskitError('Counts objects with dit strings do not currently support conversion to hexadecimal')
            int_key = self._remove_space_underscore(bitstring)
            out_dict[hex(int_key)] = value
        return out_dict