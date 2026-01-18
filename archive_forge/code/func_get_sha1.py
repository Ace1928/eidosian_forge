import os
import stat
import time
from .. import atomicfile, errors
from .. import filters as _mod_filters
from .. import osutils, trace
def get_sha1(self, path, stat_value=None):
    """Return the sha1 of a file.
        """
    abspath = osutils.pathjoin(self.root, path)
    self.stat_count += 1
    file_fp = self._fingerprint(abspath, stat_value)
    if not file_fp:
        if path in self._cache:
            self.removed_count += 1
            self.needs_write = True
            del self._cache[path]
        return None
    if path in self._cache:
        cache_sha1, cache_fp = self._cache[path]
    else:
        cache_sha1, cache_fp = (None, None)
    if cache_fp == file_fp:
        self.hit_count += 1
        return cache_sha1
    self.miss_count += 1
    mode = file_fp[FP_MODE_COLUMN]
    if stat.S_ISREG(mode):
        if self._filter_provider is None:
            filters = []
        else:
            filters = self._filter_provider(path=path)
        digest = self._really_sha1_file(abspath, filters)
    elif stat.S_ISLNK(mode):
        target = osutils.readlink(abspath)
        digest = osutils.sha_string(target.encode('UTF-8'))
    else:
        raise errors.BzrError('file %r: unknown file stat mode: %o' % (abspath, mode))
    cutoff = self._cutoff_time()
    if file_fp[FP_MTIME_COLUMN] >= cutoff or file_fp[FP_CTIME_COLUMN] >= cutoff:
        self.danger_count += 1
        if cache_fp:
            self.removed_count += 1
            self.needs_write = True
            del self._cache[path]
    else:
        self.update_count += 1
        self.needs_write = True
        self._cache[path] = (digest, file_fp)
    return digest