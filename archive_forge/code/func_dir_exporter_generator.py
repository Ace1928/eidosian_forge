import errno
import os
import sys
import time
from . import archive, errors, osutils, trace
def dir_exporter_generator(tree, dest, root, subdir=None, force_mtime=None, fileobj=None, recurse_nested=False):
    """Return a generator that exports this tree to a new directory.

    `dest` should either not exist or should be empty. If it does not exist it
    will be created holding the contents of this tree.

    :note: If the export fails, the destination directory will be
           left in an incompletely exported state: export is not transactional.
    """
    try:
        os.mkdir(dest)
    except OSError as e:
        if e.errno == errno.EEXIST:
            if os.listdir(dest) != []:
                raise errors.BzrError("Can't export tree to non-empty directory.")
        else:
            raise
    to_fetch = []
    for dp, tp, ie in _export_iter_entries(tree, subdir, recurse_nested=recurse_nested):
        fullpath = osutils.pathjoin(dest, dp)
        if ie.kind == 'file':
            to_fetch.append((tp, (dp, tp, None)))
        elif ie.kind in ('directory', 'tree-reference'):
            os.mkdir(fullpath)
        elif ie.kind == 'symlink':
            try:
                symlink_target = tree.get_symlink_target(tp)
                os.symlink(symlink_target, fullpath)
            except OSError as e:
                raise errors.BzrError('Failed to create symlink %r -> %r, error: %s' % (fullpath, symlink_target, e))
        else:
            raise errors.BzrError("don't know how to export {%s} of kind %r" % (tp, ie.kind))
        yield
    flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY | getattr(os, 'O_BINARY', 0)
    for (relpath, treepath, unused_none), chunks in tree.iter_files_bytes(to_fetch):
        fullpath = osutils.pathjoin(dest, relpath)
        mode = 438
        if tree.is_executable(treepath):
            mode = 511
        with os.fdopen(os.open(fullpath, flags, mode), 'wb') as out:
            out.writelines(chunks)
        if force_mtime is not None:
            mtime = force_mtime
        else:
            mtime = tree.get_file_mtime(treepath)
        os.utime(fullpath, (mtime, mtime))
        yield