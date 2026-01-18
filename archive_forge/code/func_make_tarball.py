import os
from warnings import warn
import sys
from distutils.errors import DistutilsExecError
from distutils.spawn import spawn
from distutils.dir_util import mkpath
from distutils import log
def make_tarball(base_name, base_dir, compress='gzip', verbose=0, dry_run=0, owner=None, group=None):
    """Create a (possibly compressed) tar file from all the files under
    'base_dir'.

    'compress' must be "gzip" (the default), "bzip2", "xz", "compress", or
    None.  ("compress" will be deprecated in Python 3.2)

    'owner' and 'group' can be used to define an owner and a group for the
    archive that is being built. If not provided, the current owner and group
    will be used.

    The output tar file will be named 'base_dir' +  ".tar", possibly plus
    the appropriate compression extension (".gz", ".bz2", ".xz" or ".Z").

    Returns the output filename.
    """
    tar_compression = {'gzip': 'gz', 'bzip2': 'bz2', 'xz': 'xz', None: '', 'compress': ''}
    compress_ext = {'gzip': '.gz', 'bzip2': '.bz2', 'xz': '.xz', 'compress': '.Z'}
    if compress is not None and compress not in compress_ext.keys():
        raise ValueError("bad value for 'compress': must be None, 'gzip', 'bzip2', 'xz' or 'compress'")
    archive_name = base_name + '.tar'
    if compress != 'compress':
        archive_name += compress_ext.get(compress, '')
    mkpath(os.path.dirname(archive_name), dry_run=dry_run)
    import tarfile
    log.info('Creating tar archive')
    uid = _get_uid(owner)
    gid = _get_gid(group)

    def _set_uid_gid(tarinfo):
        if gid is not None:
            tarinfo.gid = gid
            tarinfo.gname = group
        if uid is not None:
            tarinfo.uid = uid
            tarinfo.uname = owner
        return tarinfo
    if not dry_run:
        tar = tarfile.open(archive_name, 'w|%s' % tar_compression[compress])
        try:
            tar.add(base_dir, filter=_set_uid_gid)
        finally:
            tar.close()
    if compress == 'compress':
        warn("'compress' will be deprecated.", PendingDeprecationWarning)
        compressed_name = archive_name + compress_ext[compress]
        if sys.platform == 'win32':
            cmd = [compress, archive_name, compressed_name]
        else:
            cmd = [compress, '-f', archive_name]
        spawn(cmd, dry_run=dry_run)
        return compressed_name
    return archive_name