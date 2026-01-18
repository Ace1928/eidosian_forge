import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def pi(self, target, data):
    if self._ignored_depth:
        return
    if self._root_done:
        self._write('\n')
    elif self._root_seen and self._data:
        self._flush()
    self._write(f'<?{target} {_escape_cdata_c14n(data)}?>' if data else f'<?{target}?>')
    if not self._root_seen:
        self._write('\n')