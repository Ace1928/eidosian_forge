import contextlib
import dataclasses
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Generic, Iterable, List, Optional, TypeVar, Union
import catalogue
import confection
@my_registry.cats('int_cat.v1')
def int_cat(value_in: Optional[int]=None, value_out: Optional[int]=None) -> Cat[Optional[int], Optional[int]]:
    """Instantiates cat with integer values."""
    return Cat(name='int_cat', value_in=value_in, value_out=value_out)