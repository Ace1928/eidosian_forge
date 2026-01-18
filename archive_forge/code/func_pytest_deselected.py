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
def pytest_deselected(items: Sequence['Item']) -> None:
    """Called for deselected test items, e.g. by keyword.

    May be called multiple times.

    :param items:
        The items.

    Use in conftest plugins
    =======================

    Any conftest file can implement this hook.
    """