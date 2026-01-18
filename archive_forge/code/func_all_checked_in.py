from typing import Any, Callable, List, Optional, Union
import torch
@property
def all_checked_in(self) -> bool:
    """Have all the expected gradient check-in happened ?"""
    return len(self._params) == self.params_checked_in