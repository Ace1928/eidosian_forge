import re
def binary_probabilities(self, num_bits=None):
    """Build a probabilities dictionary with binary string keys

        Parameters:
            num_bits (int): number of bits in the binary bitstrings (leading
                zeros will be padded). If None, a default value will be used.
                If keys are given as integers or strings with binary or hex prefix,
                the default value will be derived from the largest key present.
                If keys are given as bitstrings without prefix,
                the default value will be derived from the largest key length.

        Returns:
            dict: A dictionary where the keys are binary strings in the format
                ``"0110"``
        """
    n = self._num_bits if num_bits is None else num_bits
    return {format(key, 'b').zfill(n): value for key, value in self.items()}