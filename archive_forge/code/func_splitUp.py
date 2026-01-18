import re
import distutils.version
import operator
def splitUp(pred):
    """Parse a single version comparison.

    Return (comparison string, StrictVersion)
    """
    res = re_splitComparison.match(pred)
    if not res:
        raise ValueError('bad package restriction syntax: %r' % pred)
    comp, verStr = res.groups()
    return (comp, distutils.version.StrictVersion(verStr))