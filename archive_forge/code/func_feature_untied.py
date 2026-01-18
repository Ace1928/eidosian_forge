import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_untied(self, names):
    """
        same as 'feature_ahead()' but if both features implied each other
        and keep the highest interest.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41"])
        ["SSE2", "SSE3", "SSE41"]
        # assume AVX2 and FMA3 implies each other
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41", "FMA3", "AVX2"])
        ["SSE2", "SSE3", "SSE41", "AVX2"]
        """
    assert not isinstance(names, str) and hasattr(names, '__iter__')
    final = []
    for n in names:
        implies = self.feature_implies(n)
        tied = [nn for nn in final if nn in implies and n in self.feature_implies(nn)]
        if tied:
            tied = self.feature_sorted(tied + [n])
            if n not in tied[1:]:
                continue
            final.remove(tied[:1][0])
        final.append(n)
    return final