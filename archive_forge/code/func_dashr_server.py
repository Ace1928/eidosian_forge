import pytest
from .consts import SELENIUM_GRID_DEFAULT
@pytest.fixture
def dashr_server() -> RRunner:
    with RRunner() as starter:
        yield starter