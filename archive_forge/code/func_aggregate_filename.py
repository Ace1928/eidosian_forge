import os
import os.path
import warnings
from ..base import CommandLine
def aggregate_filename(files, new_suffix):
    """
    Try to work out a sensible name given a set of files that have
    been combined in some way (e.g. averaged). If we can't work out a
    sensible prefix, we use the first filename in the list.

    Examples
    --------

    >>> from nipype.interfaces.minc.base import aggregate_filename
    >>> f = aggregate_filename(['/tmp/foo1.mnc', '/tmp/foo2.mnc', '/tmp/foo3.mnc'], 'averaged')
    >>> os.path.split(f)[1] # This has a full path, so just check the filename.
    'foo_averaged.mnc'

    >>> f = aggregate_filename(['/tmp/foo1.mnc', '/tmp/blah1.mnc'], 'averaged')
    >>> os.path.split(f)[1] # This has a full path, so just check the filename.
    'foo1_averaged.mnc'

    """
    path = os.path.split(files[0])[0]
    names = [os.path.splitext(os.path.split(x)[1])[0] for x in files]
    common_prefix = os.path.commonprefix(names)
    path = os.getcwd()
    if common_prefix == '':
        return os.path.abspath(os.path.join(path, os.path.splitext(files[0])[0] + '_' + new_suffix + '.mnc'))
    else:
        return os.path.abspath(os.path.join(path, common_prefix + '_' + new_suffix + '.mnc'))