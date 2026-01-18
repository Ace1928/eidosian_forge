import fnmatch
import os
import os.path
import random
import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Generator, List, Union, cast
from pip._vendor.tenacity import retry, stop_after_delay, wait_fixed
from pip._internal.utils.compat import get_path_uid
from pip._internal.utils.misc import format_size
@contextmanager
def adjacent_tmp_file(path: str, **kwargs: Any) -> Generator[BinaryIO, None, None]:
    """Return a file-like object pointing to a tmp file next to path.

    The file is created securely and is ensured to be written to disk
    after the context reaches its end.

    kwargs will be passed to tempfile.NamedTemporaryFile to control
    the way the temporary file will be opened.
    """
    with NamedTemporaryFile(delete=False, dir=os.path.dirname(path), prefix=os.path.basename(path), suffix='.tmp', **kwargs) as f:
        result = cast(BinaryIO, f)
        try:
            yield result
        finally:
            result.flush()
            os.fsync(result.fileno())