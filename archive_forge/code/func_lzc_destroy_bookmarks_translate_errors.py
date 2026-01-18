import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_destroy_bookmarks_translate_errors(ret, errlist, bookmarks):
    if ret == 0:
        return

    def _map(ret, name):
        if ret == errno.EINVAL:
            return lzc_exc.NameInvalid(name)
        return _generic_exception(ret, name, 'Failed to destroy bookmark')
    _handle_err_list(ret, errlist, bookmarks, lzc_exc.BookmarkDestructionFailure, _map)