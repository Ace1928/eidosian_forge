import inspect
from functools import partial
from typing import (
def rich_repr(cls: Optional[Type[T]]=None, *, angular: bool=False) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    if cls is None:
        return auto(angular=angular)
    else:
        return auto(cls)