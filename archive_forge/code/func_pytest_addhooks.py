import pytest
from .consts import SELENIUM_GRID_DEFAULT
@pytest.mark.tryfirst
def pytest_addhooks(pluginmanager):
    if not _installed:
        return
    from dash.testing import newhooks
    method = getattr(pluginmanager, 'add_hookspecs', None)
    if method is None:
        method = pluginmanager.addhooks
    method(newhooks)