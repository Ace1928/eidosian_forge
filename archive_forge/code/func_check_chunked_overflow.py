import os
from pyarrow.pandas_compat import _pandas_api  # noqa
from pyarrow.lib import (Codec, Table,  # noqa
import pyarrow.lib as ext
from pyarrow import _feather
from pyarrow._feather import FeatherError  # noqa: F401
def check_chunked_overflow(name, col):
    if col.num_chunks == 1:
        return
    if col.type in (ext.binary(), ext.string()):
        raise ValueError("Column '{}' exceeds 2GB maximum capacity of a Feather binary column. This restriction may be lifted in the future".format(name))
    else:
        raise ValueError("Column '{}' of type {} was chunked on conversion to Arrow and cannot be currently written to Feather format".format(name, str(col.type)))