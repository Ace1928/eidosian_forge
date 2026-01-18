import sys
import re
import joblib
def get_joblib_version(joblib_version=joblib.__version__):
    """Normalize joblib version by removing suffix.

    >>> get_joblib_version('0.8.4')
    '0.8.4'
    >>> get_joblib_version('0.8.4b1')
    '0.8.4'
    >>> get_joblib_version('0.9.dev0')
    '0.9'
    """
    matches = [re.match('(\\d+).*', each) for each in joblib_version.split('.')]
    return '.'.join([m.group(1) for m in matches if m is not None])