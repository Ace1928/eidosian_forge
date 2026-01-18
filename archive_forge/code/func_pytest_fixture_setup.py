from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from pluggy import HookspecMarker
@hookspec(firstresult=True)
def pytest_fixture_setup(fixturedef: 'FixtureDef[Any]', request: 'SubRequest') -> Optional[object]:
    """Perform fixture setup execution.

    :param fixturdef:
        The fixture definition object.
    :param request:
        The fixture request object.
    :returns:
        The return value of the call to the fixture function.

    Stops at first non-None result, see :ref:`firstresult`.

    .. note::
        If the fixture function returns None, other implementations of
        this hook function will continue to be called, according to the
        behavior of the :ref:`firstresult` option.

    Use in conftest plugins
    =======================

    Any conftest file can implement this hook. For a given fixture, only
    conftest files in the fixture scope's directory and its parent directories
    are consulted.
    """