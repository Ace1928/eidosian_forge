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
def pytest_make_collect_report(collector: 'Collector') -> 'Optional[CollectReport]':
    """Perform :func:`collector.collect() <pytest.Collector.collect>` and return
    a :class:`~pytest.CollectReport`.

    Stops at first non-None result, see :ref:`firstresult`.

    :param collector:
        The collector.

    Use in conftest plugins
    =======================

    Any conftest file can implement this hook. For a given collector, only
    conftest files in the collector's directory and its parent directories are
    consulted.
    """