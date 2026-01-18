import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_bookmark_translate_errors(ret, errlist, bookmarks):
    if ret == 0:
        return

    def _map(ret, name):
        if ret == errno.EINVAL:
            if name:
                snap = bookmarks[name]
                pool_names = map(_pool_name, bookmarks.keys())
                if not _is_valid_bmark_name(name):
                    return lzc_exc.BookmarkNameInvalid(name)
                elif not _is_valid_snap_name(snap):
                    return lzc_exc.SnapshotNameInvalid(snap)
                elif _fs_name(name) != _fs_name(snap):
                    return lzc_exc.BookmarkMismatch(name)
                elif any((x != _pool_name(name) for x in pool_names)):
                    return lzc_exc.PoolsDiffer(name)
            else:
                invalid_names = [b for b in bookmarks.keys() if not _is_valid_bmark_name(b)]
                if invalid_names:
                    return lzc_exc.BookmarkNameInvalid(invalid_names[0])
        if ret == errno.EEXIST:
            return lzc_exc.BookmarkExists(name)
        if ret == errno.ENOENT:
            return lzc_exc.SnapshotNotFound(name)
        if ret == errno.ENOTSUP:
            return lzc_exc.BookmarkNotSupported(name)
        return _generic_exception(ret, name, 'Failed to create bookmark')
    _handle_err_list(ret, errlist, bookmarks.keys(), lzc_exc.BookmarkFailure, _map)