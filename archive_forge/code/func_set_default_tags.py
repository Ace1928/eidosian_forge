import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from ray._raylet import (
from ray.util.annotations import DeveloperAPI
def set_default_tags(self, default_tags: Dict[str, str]):
    """Set default tags of metrics.

        Example:
            >>> from ray.util.metrics import Counter
            >>> # Note that set_default_tags returns the instance itself.
            >>> counter = Counter("name", tag_keys=("a",))
            >>> counter2 = counter.set_default_tags({"a": "b"})
            >>> assert counter is counter2
            >>> # this means you can instantiate it in this way.
            >>> counter = Counter("name", tag_keys=("a",)).set_default_tags({"a": "b"})

        Args:
            default_tags: Default tags that are
                used for every record method.

        Returns:
            Metric: it returns the instance itself.
        """
    for key, val in default_tags.items():
        if key not in self._tag_keys:
            raise ValueError(f'Unrecognized tag key {key}.')
        if not isinstance(val, str):
            raise TypeError(f'Tag values must be str, got {type(val)}.')
    self._default_tags = default_tags
    return self