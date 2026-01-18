from __future__ import annotations
import warnings
from _pytest.recwarn import WarningsChecker
from pytest import warns
def pytest_warns(warning: type[Warning] | tuple[type[Warning], ...] | None) -> WarningsChecker | NoWarningsChecker:
    """

    Parameters
    ----------
    warning : {None, Warning, Tuple[Warning]}
        None if no warning is produced, or a single or multiple Warnings

    Returns
    -------
    cm

    """
    if warning is None:
        return NoWarningsChecker()
    else:
        assert warning is not None
        return warns(warning)