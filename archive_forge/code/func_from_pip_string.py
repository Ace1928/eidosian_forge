import itertools
import operator
import sys
@classmethod
def from_pip_string(klass, version_string):
    """Create a SemanticVersion from a pip version string.

        This method will parse a version like 1.3.0 into a SemanticVersion.

        This method is responsible for accepting any version string that any
        older version of pbr ever created.

        Therefore: versions like 1.3.0a1 versions are handled, parsed into a
        canonical form and then output - resulting in 1.3.0.0a1.
        Pre pbr-semver dev versions like 0.10.1.3.g83bef74 will be parsed but
        output as 0.10.1.dev3.g83bef74.

        :raises ValueError: Never tagged versions sdisted by old pbr result in
            just the git hash, e.g. '1234567' which poses a substantial problem
            since they collide with the semver versions when all the digits are
            numerals. Such versions will result in a ValueError being thrown if
            any non-numeric digits are present. They are an exception to the
            general case of accepting anything we ever output, since they were
            never intended and would permanently mess up versions on PyPI if
            ever released - we're treating that as a critical bug that we ever
            made them and have stopped doing that.
        """
    try:
        return klass._from_pip_string_unsafe(version_string)
    except IndexError:
        raise ValueError('Invalid version %r' % version_string)