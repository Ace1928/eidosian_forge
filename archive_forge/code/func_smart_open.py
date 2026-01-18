import collections
import io
import locale
import logging
import os
import os.path as P
import pathlib
import urllib.parse
import warnings
import smart_open.local_file as so_file
import smart_open.compression as so_compression
from smart_open import doctools
from smart_open import transport
from smart_open.compression import register_compressor  # noqa: F401
from smart_open.utils import check_kwargs as _check_kwargs  # noqa: F401
from smart_open.utils import inspect_kwargs as _inspect_kwargs  # noqa: F401
def smart_open(uri, mode='rb', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None, ignore_extension=False, **kwargs):
    url = 'https://github.com/RaRe-Technologies/smart_open/blob/develop/MIGRATING_FROM_OLDER_VERSIONS.rst'
    if kwargs:
        raise DeprecationWarning('The following keyword parameters are not supported: %r. See  %s for more information.' % (sorted(kwargs), url))
    message = 'This function is deprecated.  See %s for more information' % url
    warnings.warn(message, category=DeprecationWarning)
    if ignore_extension:
        compression = so_compression.NO_COMPRESSION
    else:
        compression = so_compression.INFER_FROM_EXTENSION
    del kwargs, url, message, ignore_extension
    return open(**locals())