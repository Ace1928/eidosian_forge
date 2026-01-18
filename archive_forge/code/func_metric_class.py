from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
@property
def metric_class(self) -> Type[MonetaryMetric]:
    """
        Returns the metric class
        """
    return MonetaryMetric