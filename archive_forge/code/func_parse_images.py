import io
import re
import functools
import inspect
import os
import sys
import numbers
import warnings
from pathlib import Path, PurePath
from typing import (
from ase.atoms import Atoms
from importlib import import_module
from ase.parallel import parallel_function, parallel_generator
def parse_images(self, data: Union[str, bytes], **kwargs) -> Sequence[Atoms]:
    with self._buf_as_filelike(data) as fd:
        outputs = self.read(fd, **kwargs)
        if self.single:
            assert isinstance(outputs, Atoms)
            return [outputs]
        else:
            return list(self.read(fd, **kwargs))