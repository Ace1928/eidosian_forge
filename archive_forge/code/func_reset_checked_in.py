from typing import Any, Callable, List, Optional, Union
import torch
def reset_checked_in(self) -> None:
    """Reset the counter of the parameter grads which have been checked in"""
    self.params_checked_in = 0
    self.sent = False