import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def set_segment_parameter(self, name, value):
    """Set a segment parameter.

        Args:
          name: Segment parameter name (urlencoded string)
          value: Segment parameter value (urlencoded string)
        """
    if value is None:
        try:
            del self._segment_parameters[name]
        except KeyError:
            pass
    else:
        self._segment_parameters[name] = value
    self.base = urlutils.join_segment_parameters(self._raw_base, self._segment_parameters)