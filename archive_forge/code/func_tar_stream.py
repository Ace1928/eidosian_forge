import posixpath
import stat
import struct
import tarfile
from contextlib import closing
from io import BytesIO
from os import SEEK_END
def tar_stream(store, tree, mtime, prefix=b'', format=''):
    """Generate a tar stream for the contents of a Git tree.

    Returns a generator that lazily assembles a .tar.gz archive, yielding it in
    pieces (bytestrings). To obtain the complete .tar.gz binary file, simply
    concatenate these chunks.

    Args:
      store: Object store to retrieve objects from
      tree: Tree object for the tree root
      mtime: UNIX timestamp that is assigned as the modification time for
        all files, and the gzip header modification time if format='gz'
      format: Optional compression format for tarball
    Returns:
      Bytestrings
    """
    buf = BytesIO()
    with closing(tarfile.open(None, 'w:%s' % format, buf)) as tar:
        if format == 'gz':
            buf.seek(0)
            assert buf.read(2) == b'\x1f\x8b', 'Invalid gzip header'
            buf.seek(4)
            buf.write(struct.pack('<L', mtime))
            buf.seek(0, SEEK_END)
        for entry_abspath, entry in _walk_tree(store, tree, prefix):
            try:
                blob = store[entry.sha]
            except KeyError:
                continue
            data = ChunkedBytesIO(blob.chunked)
            info = tarfile.TarInfo()
            info.name = entry_abspath.decode('utf-8', 'surrogateescape')
            info.size = blob.raw_length()
            info.mode = entry.mode
            info.mtime = mtime
            tar.addfile(info, data)
            yield buf.getvalue()
            buf.truncate(0)
            buf.seek(0)
    yield buf.getvalue()