from warnings import warn
from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestRemovedIn9Warning
from _pytest.warning_types import UnformattedWarning
def check_ispytest(ispytest: bool) -> None:
    if not ispytest:
        warn(PRIVATE, stacklevel=3)