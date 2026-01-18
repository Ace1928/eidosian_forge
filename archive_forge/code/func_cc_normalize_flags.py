import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def cc_normalize_flags(self, flags):
    """
        Remove the conflicts that caused due gathering implied features flags.

        Parameters
        ----------
        'flags' list, compiler flags
            flags should be sorted from the lowest to the highest interest.

        Returns
        -------
        list, filtered from any conflicts.

        Examples
        --------
        >>> self.cc_normalize_flags(['-march=armv8.2-a+fp16', '-march=armv8.2-a+dotprod'])
        ['armv8.2-a+fp16+dotprod']

        >>> self.cc_normalize_flags(
            ['-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1', '-msse4.2', '-mavx', '-march=core-avx2']
        )
        ['-march=core-avx2']
        """
    assert isinstance(flags, list)
    if self.cc_is_gcc or self.cc_is_clang or self.cc_is_icc:
        return self._cc_normalize_unix(flags)
    if self.cc_is_msvc or self.cc_is_iccw:
        return self._cc_normalize_win(flags)
    return flags