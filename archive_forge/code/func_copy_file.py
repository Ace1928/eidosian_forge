import os
from .errors import DistutilsFileError
from ._log import log
def copy_file(src, dst, preserve_mode=1, preserve_times=1, update=0, link=None, verbose=1, dry_run=0):
    """Copy a file 'src' to 'dst'.  If 'dst' is a directory, then 'src' is
    copied there with the same name; otherwise, it must be a filename.  (If
    the file exists, it will be ruthlessly clobbered.)  If 'preserve_mode'
    is true (the default), the file's mode (type and permission bits, or
    whatever is analogous on the current platform) is copied.  If
    'preserve_times' is true (the default), the last-modified and
    last-access times are copied as well.  If 'update' is true, 'src' will
    only be copied if 'dst' does not exist, or if 'dst' does exist but is
    older than 'src'.

    'link' allows you to make hard links (os.link) or symbolic links
    (os.symlink) instead of copying: set it to "hard" or "sym"; if it is
    None (the default), files are copied.  Don't set 'link' on systems that
    don't support it: 'copy_file()' doesn't check if hard or symbolic
    linking is available. If hardlink fails, falls back to
    _copy_file_contents().

    Under Mac OS, uses the native file copy function in macostools; on
    other systems, uses '_copy_file_contents()' to copy file contents.

    Return a tuple (dest_name, copied): 'dest_name' is the actual name of
    the output file, and 'copied' is true if the file was copied (or would
    have been copied, if 'dry_run' true).
    """
    from distutils._modified import newer
    from stat import ST_ATIME, ST_MTIME, ST_MODE, S_IMODE
    if not os.path.isfile(src):
        raise DistutilsFileError("can't copy '%s': doesn't exist or not a regular file" % src)
    if os.path.isdir(dst):
        dir = dst
        dst = os.path.join(dst, os.path.basename(src))
    else:
        dir = os.path.dirname(dst)
    if update and (not newer(src, dst)):
        if verbose >= 1:
            log.debug('not copying %s (output up-to-date)', src)
        return (dst, 0)
    try:
        action = _copy_action[link]
    except KeyError:
        raise ValueError("invalid value '%s' for 'link' argument" % link)
    if verbose >= 1:
        if os.path.basename(dst) == os.path.basename(src):
            log.info('%s %s -> %s', action, src, dir)
        else:
            log.info('%s %s -> %s', action, src, dst)
    if dry_run:
        return (dst, 1)
    elif link == 'hard':
        if not (os.path.exists(dst) and os.path.samefile(src, dst)):
            try:
                os.link(src, dst)
                return (dst, 1)
            except OSError:
                pass
    elif link == 'sym':
        if not (os.path.exists(dst) and os.path.samefile(src, dst)):
            os.symlink(src, dst)
            return (dst, 1)
    _copy_file_contents(src, dst)
    if preserve_mode or preserve_times:
        st = os.stat(src)
        if preserve_times:
            os.utime(dst, (st[ST_ATIME], st[ST_MTIME]))
        if preserve_mode:
            os.chmod(dst, S_IMODE(st[ST_MODE]))
    return (dst, 1)