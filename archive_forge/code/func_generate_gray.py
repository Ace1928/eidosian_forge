from sympy.core import Basic, Integer
import random
def generate_gray(self, **hints):
    """
        Generates the sequence of bit vectors of a Gray Code.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> list(a.generate_gray())
        ['000', '001', '011', '010', '110', '111', '101', '100']
        >>> list(a.generate_gray(start='011'))
        ['011', '010', '110', '111', '101', '100']
        >>> list(a.generate_gray(rank=4))
        ['110', '111', '101', '100']

        See Also
        ========

        skip

        References
        ==========

        .. [1] Knuth, D. (2011). The Art of Computer Programming,
               Vol 4, Addison Wesley

        """
    bits = self.n
    start = None
    if 'start' in hints:
        start = hints['start']
    elif 'rank' in hints:
        start = GrayCode.unrank(self.n, hints['rank'])
    if start is not None:
        self._current = start
    current = self.current
    graycode_bin = gray_to_bin(current)
    if len(graycode_bin) > self.n:
        raise ValueError('Gray code start has length %i but should not be greater than %i' % (len(graycode_bin), bits))
    self._current = int(current, 2)
    graycode_int = int(''.join(graycode_bin), 2)
    for i in range(graycode_int, 1 << bits):
        if self._skip:
            self._skip = False
        else:
            yield self.current
        bbtc = i ^ i + 1
        gbtc = bbtc ^ bbtc >> 1
        self._current = self._current ^ gbtc
    self._current = 0