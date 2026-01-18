import array
import logging
import posixpath
import warnings
from collections.abc import MutableMapping
from functools import cached_property
from fsspec.core import url_to_fs
def setitems(self, values_dict):
    """Set the values of multiple items in the store

        Parameters
        ----------
        values_dict: dict(str, bytes)
        """
    values = {self._key_to_str(k): maybe_convert(v) for k, v in values_dict.items()}
    self.fs.pipe(values)