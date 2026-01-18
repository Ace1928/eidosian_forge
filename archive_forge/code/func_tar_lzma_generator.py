import os
import sys
import tarfile
from contextlib import closing
from io import BytesIO
from .. import errors, osutils
from ..export import _export_iter_entries
def tar_lzma_generator(tree, dest, root, subdir, force_mtime=None, compression_format='alone', recurse_nested=False):
    """Export this tree to a new .tar.lzma file.

    `dest` will be created holding the contents of this tree; if it
    already exists, it will be clobbered, like with "tar -c".
    """
    try:
        import lzma
    except ModuleNotFoundError as exc:
        raise errors.DependencyNotPresent('lzma', e) from exc
    compressor = lzma.LZMACompressor(format={'xz': lzma.FORMAT_XZ, 'raw': lzma.FORMAT_RAW, 'alone': lzma.FORMAT_ALONE}[compression_format])
    for chunk in tarball_generator(tree, root, subdir, force_mtime=force_mtime, recurse_nested=recurse_nested):
        yield compressor.compress(chunk)
    yield compressor.flush()