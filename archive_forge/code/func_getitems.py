import array
import logging
import posixpath
import warnings
from collections.abc import MutableMapping
from functools import cached_property
from fsspec.core import url_to_fs
def getitems(self, keys, on_error='raise'):
    """Fetch multiple items from the store

        If the backend is async-able, this might proceed concurrently

        Parameters
        ----------
        keys: list(str)
            They keys to be fetched
        on_error : "raise", "omit", "return"
            If raise, an underlying exception will be raised (converted to KeyError
            if the type is in self.missing_exceptions); if omit, keys with exception
            will simply not be included in the output; if "return", all keys are
            included in the output, but the value will be bytes or an exception
            instance.

        Returns
        -------
        dict(key, bytes|exception)
        """
    keys2 = [self._key_to_str(k) for k in keys]
    oe = on_error if on_error == 'raise' else 'return'
    try:
        out = self.fs.cat(keys2, on_error=oe)
        if isinstance(out, bytes):
            out = {keys2[0]: out}
    except self.missing_exceptions as e:
        raise KeyError from e
    out = {k: KeyError() if isinstance(v, self.missing_exceptions) else v for k, v in out.items()}
    return {key: out[k2] for key, k2 in zip(keys, keys2) if on_error == 'return' or not isinstance(out[k2], BaseException)}