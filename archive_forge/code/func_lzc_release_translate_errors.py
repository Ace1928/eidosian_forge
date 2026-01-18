import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_release_translate_errors(ret, errlist, holds):
    if ret == 0:
        return
    for _, hold_list in holds.iteritems():
        if not isinstance(hold_list, list):
            raise lzc_exc.TypeError('holds must be in a list')

    def _map(ret, name):
        if ret == errno.EXDEV:
            return lzc_exc.PoolsDiffer(name)
        elif ret == errno.EINVAL:
            if name:
                pool_names = map(_pool_name, holds.keys())
                if not _is_valid_snap_name(name):
                    return lzc_exc.NameInvalid(name)
                elif len(name) > MAXNAMELEN:
                    return lzc_exc.NameTooLong(name)
                elif any((x != _pool_name(name) for x in pool_names)):
                    return lzc_exc.PoolsDiffer(name)
            else:
                invalid_names = [b for b in holds.keys() if not _is_valid_snap_name(b)]
                if invalid_names:
                    return lzc_exc.NameInvalid(invalid_names[0])
        elif ret == errno.ENOENT:
            return lzc_exc.HoldNotFound(name)
        elif ret == errno.E2BIG:
            tag_list = holds[name]
            too_long_tags = [t for t in tag_list if len(t) > MAXNAMELEN]
            return lzc_exc.NameTooLong(too_long_tags[0])
        elif ret == errno.ENOTSUP:
            pool_name = None
            if name is not None:
                pool_name = _pool_name(name)
            return lzc_exc.FeatureNotSupported(pool_name)
        else:
            return _generic_exception(ret, name, 'Failed to release snapshot hold')
    _handle_err_list(ret, errlist, holds.keys(), lzc_exc.HoldReleaseFailure, _map)