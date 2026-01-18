import os
import sys
import tarfile
from contextlib import closing
from io import BytesIO
from .. import errors, osutils
from ..export import _export_iter_entries
def tar_xz_generator(tree, dest, root, subdir, force_mtime=None, recurse_nested=False):
    return tar_lzma_generator(tree, dest, root, subdir, force_mtime, 'xz', recurse_nested=recurse_nested)