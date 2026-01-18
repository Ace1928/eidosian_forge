import os, stat
def get_runtime_dir(strict=True):
    """Returns the value of $XDG_RUNTIME_DIR, a directory path.
    
    This directory is intended for 'user-specific non-essential runtime files
    and other file objects (such as sockets, named pipes, ...)', and
    'communication and synchronization purposes'.
    
    As of late 2012, only quite new systems set $XDG_RUNTIME_DIR. If it is not
    set, with ``strict=True`` (the default), a KeyError is raised. With 
    ``strict=False``, PyXDG will create a fallback under /tmp for the current
    user. This fallback does *not* provide the same guarantees as the
    specification requires for the runtime directory.
    
    The strict default is deliberately conservative, so that application
    developers can make a conscious decision to allow the fallback.
    """
    try:
        return os.environ['XDG_RUNTIME_DIR']
    except KeyError:
        if strict:
            raise
        import getpass
        fallback = '/tmp/pyxdg-runtime-dir-fallback-' + getpass.getuser()
        create = False
        try:
            st = os.lstat(fallback)
        except OSError as e:
            import errno
            if e.errno == errno.ENOENT:
                create = True
            else:
                raise
        else:
            if not stat.S_ISDIR(st.st_mode):
                os.unlink(fallback)
                create = True
            elif st.st_uid != os.getuid() or st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                os.rmdir(fallback)
                create = True
        if create:
            os.mkdir(fallback, 448)
        return fallback