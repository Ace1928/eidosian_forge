import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def should_skip_file(name):
    """
    Checks if a file should be skipped based on its name.

    If it should be skipped, returns the reason, otherwise returns
    None.
    """
    if name.startswith('.'):
        return 'Skipping hidden file %(filename)s'
    if name.endswith('~') or name.endswith('.bak'):
        return 'Skipping backup file %(filename)s'
    if name.endswith('.pyc') or name.endswith('.pyo'):
        return 'Skipping %s file %%(filename)s' % os.path.splitext(name)[1]
    if name.endswith('$py.class'):
        return 'Skipping $py.class file %(filename)s'
    if name in ('CVS', '_darcs'):
        return 'Skipping version control directory %(filename)s'
    return None