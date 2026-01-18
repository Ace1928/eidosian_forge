import os
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp
@contextmanager
def in_dir(dir=None):
    """ Change directory to given directory for duration of ``with`` block

    Useful when you want to use `in_tempdir` for the final test, but
    you are still debugging. For example, you may want to do this in the end:

    >>> with in_tempdir() as tmpdir:
    ...     # do something complicated which might break
    ...     pass

    But, indeed, the complicated thing does break, and meanwhile, the
    ``in_tempdir`` context manager wiped out the directory with the
    temporary files that you wanted for debugging. So, while debugging, you
    replace with something like:

    >>> with in_dir() as tmpdir: # Use working directory by default
    ...     # do something complicated which might break
    ...     pass

    You can then look at the temporary file outputs to debug what is happening,
    fix, and finally replace ``in_dir`` with ``in_tempdir`` again.
    """
    cwd = os.getcwd()
    if dir is None:
        yield cwd
        return
    os.chdir(dir)
    yield dir
    os.chdir(cwd)