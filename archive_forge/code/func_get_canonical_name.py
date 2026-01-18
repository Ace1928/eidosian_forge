from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def get_canonical_name(api_names: Sequence[str], deprecated_api_names: Sequence[str]) -> Optional[str]:
    """Get preferred endpoint name.

  Args:
    api_names: API names iterable.
    deprecated_api_names: Deprecated API names iterable.

  Returns:
    Returns one of the following in decreasing preference:
    - first non-deprecated endpoint
    - first endpoint
    - None
  """
    non_deprecated_name = next((name for name in api_names if name not in deprecated_api_names), None)
    if non_deprecated_name:
        return non_deprecated_name
    if api_names:
        return api_names[0]
    return None