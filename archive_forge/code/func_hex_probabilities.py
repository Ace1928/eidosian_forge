import re
def hex_probabilities(self):
    """Build a probabilities dictionary with hexadecimal string keys

        Returns:
            dict: A dictionary where the keys are hexadecimal strings in the
                format ``"0x1a"``
        """
    return {hex(key): value for key, value in self.items()}