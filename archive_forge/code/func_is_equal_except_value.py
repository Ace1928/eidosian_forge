import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub.utils import logging, yaml_dump
def is_equal_except_value(self, other: 'EvalResult') -> bool:
    """
        Return True if `self` and `other` describe exactly the same metric but with a
        different value.
        """
    for key, _ in self.__dict__.items():
        if key == 'metric_value':
            continue
        if key != 'verify_token' and getattr(self, key) != getattr(other, key):
            return False
    return True