import itertools
import operator
import sys
def rpm_string(self):
    """Return the version number to use when building an RPM package.

        This translates the PEP440/semver precedence rules into RPM version
        sorting operators. Because RPM has no sort-before operator (such as the
        ~ operator in dpkg),  we show all prerelease versions as being versions
        of the release before.
        """
    return self._long_version(None)