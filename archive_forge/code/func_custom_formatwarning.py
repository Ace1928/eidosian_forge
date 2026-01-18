import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union
from ._version import get_versions
def custom_formatwarning(message: Union[Warning, str], category: Type[Warning], filename: str, lineno: int, line: Optional[str]=None) -> str:
    return '{}: {}\n'.format(category.__name__, message)