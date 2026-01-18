import itertools
import operator
import sys
def debian_string(self):
    """Return the version number to use when building a debian package.

        This translates the PEP440/semver precedence rules into Debian version
        sorting operators.
        """
    return self._long_version('~')