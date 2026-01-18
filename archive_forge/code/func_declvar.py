from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING
import numpy
import numpy.typing as npt
import cupy
from cupy._core._scalar import get_typename
def declvar(self, x: str, init: Optional['Data']) -> str:
    assert init is None
    if self._alignment is not None:
        code = f'__align__({self._alignment})'
    else:
        code = ''
    if self._size is None:
        code = f'extern {code} __shared__ {self.child_type} {x}[]'
    else:
        code = f'{code} __shared__ {self.child_type} {x}[{self._size}]'
    return code