import os
from .errors import DistutilsFileError
from ._log import log
def move_file(src, dst, verbose=1, dry_run=0):
    """Move a file 'src' to 'dst'.  If 'dst' is a directory, the file will
    be moved into it with the same name; otherwise, 'src' is just renamed
    to 'dst'.  Return the new full name of the file.

    Handles cross-device moves on Unix using 'copy_file()'.  What about
    other systems???
    """
    from os.path import exists, isfile, isdir, basename, dirname
    import errno
    if verbose >= 1:
        log.info('moving %s -> %s', src, dst)
    if dry_run:
        return dst
    if not isfile(src):
        raise DistutilsFileError("can't move '%s': not a regular file" % src)
    if isdir(dst):
        dst = os.path.join(dst, basename(src))
    elif exists(dst):
        raise DistutilsFileError("can't move '{}': destination '{}' already exists".format(src, dst))
    if not isdir(dirname(dst)):
        raise DistutilsFileError("can't move '{}': destination '{}' not a valid path".format(src, dst))
    copy_it = False
    try:
        os.rename(src, dst)
    except OSError as e:
        num, msg = e.args
        if num == errno.EXDEV:
            copy_it = True
        else:
            raise DistutilsFileError("couldn't move '{}' to '{}': {}".format(src, dst, msg))
    if copy_it:
        copy_file(src, dst, verbose=verbose)
        try:
            os.unlink(src)
        except OSError as e:
            num, msg = e.args
            try:
                os.unlink(dst)
            except OSError:
                pass
            raise DistutilsFileError("couldn't move '%s' to '%s' by copy/delete: delete '%s' failed: %s" % (src, dst, src, msg))
    return dst