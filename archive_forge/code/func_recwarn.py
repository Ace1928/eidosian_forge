from pprint import pformat
import re
from types import TracebackType
from typing import Any
from typing import Callable
from typing import final
from typing import Generator
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import warnings
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.outcomes import Exit
from _pytest.outcomes import fail
@fixture
def recwarn() -> Generator['WarningsRecorder', None, None]:
    """Return a :class:`WarningsRecorder` instance that records all warnings emitted by test functions.

    See https://docs.pytest.org/en/latest/how-to/capture-warnings.html for information
    on warning categories.
    """
    wrec = WarningsRecorder(_ispytest=True)
    with wrec:
        warnings.simplefilter('default')
        yield wrec