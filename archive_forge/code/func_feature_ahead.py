import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_ahead(self, names):
    """
        Return list of features in 'names' after remove any
        implied features and keep the origins.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41"])
        ["SSE41"]
        # assume AVX2 and FMA3 implies each other and AVX2
        # is the highest interest
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2"]
        # assume AVX2 and FMA3 don't implies each other
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2", "FMA3"]
        """
    assert not isinstance(names, str) and hasattr(names, '__iter__')
    implies = self.feature_implies(names, keep_origins=True)
    ahead = [n for n in names if n not in implies]
    if len(ahead) == 0:
        ahead = self.feature_sorted(names, reverse=True)[:1]
    return ahead