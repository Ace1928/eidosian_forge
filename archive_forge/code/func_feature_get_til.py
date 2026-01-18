import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_get_til(self, names, keyisfalse):
    """
        same as `feature_implies_c()` but stop collecting implied
        features when feature's option that provided through
        parameter 'keyisfalse' is False, also sorting the returned
        features.
        """

    def til(tnames):
        tnames = self.feature_implies_c(tnames)
        tnames = self.feature_sorted(tnames, reverse=True)
        for i, n in enumerate(tnames):
            if not self.feature_supported[n].get(keyisfalse, True):
                tnames = tnames[:i + 1]
                break
        return tnames
    if isinstance(names, str) or len(names) <= 1:
        names = til(names)
        names.reverse()
        return names
    names = self.feature_ahead(names)
    names = {t for n in names for t in til(n)}
    return self.feature_sorted(names)