import os
import posixpath
import sys
import urllib.parse
import warnings
from os.path import join as pjoin
import pyarrow as pa
from pyarrow.util import doc, _stringify_path, _is_path_like, _DEPR_MSG
def resolve_filesystem_and_path(where, filesystem=None):
    """
    Return filesystem from path which could be an HDFS URI, a local URI,
    or a plain filesystem path.
    """
    if not _is_path_like(where):
        if filesystem is not None:
            raise ValueError('filesystem passed but where is file-like, so there is nothing to open with filesystem.')
        return (filesystem, where)
    if filesystem is not None:
        filesystem = _ensure_filesystem(filesystem)
        if isinstance(filesystem, LocalFileSystem):
            path = _stringify_path(where)
        elif not isinstance(where, str):
            raise TypeError('Expected string path; path-like objects are only allowed with a local filesystem')
        else:
            path = where
        return (filesystem, path)
    path = _stringify_path(where)
    parsed_uri = urllib.parse.urlparse(path)
    if parsed_uri.scheme == 'hdfs' or parsed_uri.scheme == 'viewfs':
        netloc_split = parsed_uri.netloc.split(':')
        host = netloc_split[0]
        if host == '':
            host = 'default'
        else:
            host = parsed_uri.scheme + '://' + host
        port = 0
        if len(netloc_split) == 2 and netloc_split[1].isnumeric():
            port = int(netloc_split[1])
        fs = pa.hdfs._connect(host=host, port=port)
        fs_path = parsed_uri.path
    elif parsed_uri.scheme == 'file':
        fs = LocalFileSystem._get_instance()
        fs_path = parsed_uri.path
    else:
        fs = LocalFileSystem._get_instance()
        fs_path = path
    return (fs, fs_path)