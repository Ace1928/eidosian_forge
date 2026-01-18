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
def pytest_report_from_serializable(config: 'Config', data: Dict[str, Any]) -> Optional[Union['CollectReport', 'TestReport']]:
    """Restore a report object previously serialized with
    :hook:`pytest_report_to_serializable`.

    :param config: The pytest config object.

    Use in conftest plugins
    =======================

    Any conftest file can implement this hook. The exact details may depend
    on the plugin which calls the hook.
    """