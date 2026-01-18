from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
    assert not isinstance(self.comparison_value, (list, str)), "Comparison value for '$lte' operation must be a float or int."
    return {self.field_name: {'$lte': self.comparison_value}}