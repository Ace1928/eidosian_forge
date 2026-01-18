import functools
import itertools
from typing import Any, NoReturn, Optional, Union, TYPE_CHECKING
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
def call_const(self, env: 'Environment', *args: Any, **kwarg: Any) -> Expr:
    raise NotImplementedError