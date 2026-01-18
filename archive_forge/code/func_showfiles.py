from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
def showfiles(filenames: typing.Tuple[str, ...], headers: typing.Tuple[str, ...], wtitle: typing.Optional[str], pager: typing.Optional[str]) -> None:
    """R showing files.

    :param filenames: A tuple of file names.
    :param headers: A tuple of strings (TODO: check what it is)
    :wtitle: Title of the "window" showing the files.
    :pager: Pager to use to show the list of files.
    """
    for fn in filenames:
        print('File: %s' % fn)
        with open(fn) as fh:
            for row in fh:
                print(row)
            print('---')