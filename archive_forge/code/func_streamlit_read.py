from __future__ import annotations
import contextlib
import errno
import io
import os
from pathlib import Path
from streamlit import env_util, util
from streamlit.string_util import is_binary_string
@contextlib.contextmanager
def streamlit_read(path, binary=False):
    """Opens a context to read this file relative to the streamlit path.

    For example:

    with streamlit_read('foo.txt') as foo:
        ...

    opens the file `.streamlit/foo.txt`

    path   - the path to write to (within the streamlit directory)
    binary - set to True for binary IO
    """
    filename = get_streamlit_file_path(path)
    if os.stat(filename).st_size == 0:
        raise util.Error('Read zero byte file: "%s"' % filename)
    mode = 'r'
    if binary:
        mode += 'b'
    with open(os.path.join(CONFIG_FOLDER_NAME, path), mode) as handle:
        yield handle