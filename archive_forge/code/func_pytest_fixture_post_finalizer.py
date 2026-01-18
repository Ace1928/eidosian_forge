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
def pytest_fixture_post_finalizer(fixturedef: 'FixtureDef[Any]', request: 'SubRequest') -> None:
    """Called after fixture teardown, but before the cache is cleared, so
    the fixture result ``fixturedef.cached_result`` is still available (not
    ``None``).

    :param fixturdef:
        The fixture definition object.
    :param request:
        The fixture request object.

    Use in conftest plugins
    =======================

    Any conftest file can implement this hook. For a given fixture, only
    conftest files in the fixture scope's directory and its parent directories
    are consulted.
    """