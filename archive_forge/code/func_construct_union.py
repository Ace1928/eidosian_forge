import sys
from os import PathLike
from typing import TYPE_CHECKING
from typing import List, Dict, AnyStr, Any, Set
from typing import Optional, Union, Tuple, Mapping, Sequence, TypeVar, Type
from typing import Callable, Coroutine, Generator, AsyncGenerator, IO, Iterable, Iterator, AsyncIterator
from typing import cast, overload
from enum import Enum, EnumMeta
from functools import singledispatchmethod
def construct_union(modules: List):
    eval_string = ', '.join([f'{i.__module__}.{i.__name__}' for i in modules])
    return eval(f'Union[{eval_string}]')