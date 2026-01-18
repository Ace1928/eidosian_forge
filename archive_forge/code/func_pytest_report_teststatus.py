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
def pytest_report_teststatus(report: Union['CollectReport', 'TestReport'], config: 'Config') -> 'TestShortLogReport | Tuple[str, str, Union[str, Tuple[str, Mapping[str, bool]]]]':
    """Return result-category, shortletter and verbose word for status
    reporting.

    The result-category is a category in which to count the result, for
    example "passed", "skipped", "error" or the empty string.

    The shortletter is shown as testing progresses, for example ".", "s",
    "E" or the empty string.

    The verbose word is shown as testing progresses in verbose mode, for
    example "PASSED", "SKIPPED", "ERROR" or the empty string.

    pytest may style these implicitly according to the report outcome.
    To provide explicit styling, return a tuple for the verbose word,
    for example ``"rerun", "R", ("RERUN", {"yellow": True})``.

    :param report: The report object whose status is to be returned.
    :param config: The pytest config object.
    :returns: The test status.

    Stops at first non-None result, see :ref:`firstresult`.

    Use in conftest plugins
    =======================

    Any conftest plugin can implement this hook.
    """