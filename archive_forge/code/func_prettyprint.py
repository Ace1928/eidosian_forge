import functools
import re
import warnings
def prettyprint(self, indent='\t'):
    """Pretty-print the clause.
        """
    return '\n'.join(self._pretty()).replace('\t', indent)