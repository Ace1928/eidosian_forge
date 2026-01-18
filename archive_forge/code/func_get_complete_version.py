from __future__ import unicode_literals
import datetime
import os
import subprocess
def get_complete_version(version=None):
    """Returns a tuple of the promise version. If version argument is non-empty,
    then checks for correctness of the tuple provided.
    """
    if version is None:
        from promise import VERSION
        return VERSION
    else:
        assert len(version) == 5
        assert version[3] in ('alpha', 'beta', 'rc', 'final')
    return version