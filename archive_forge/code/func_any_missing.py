import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def any_missing(self):
    """Return a list of modules that appear to be missing. Use
        any_missing_maybe() if you want to know which modules are
        certain to be missing, and which *may* be missing.
        """
    missing, maybe = self.any_missing_maybe()
    return missing + maybe