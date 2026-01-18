from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from modin.logging import ClassLogger
@property
@abstractmethod
def list_of_blocks(self) -> list:
    """Get the list of physical partition objects that compose this partition."""
    pass