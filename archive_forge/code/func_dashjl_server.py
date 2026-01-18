import pytest
from .consts import SELENIUM_GRID_DEFAULT
@pytest.fixture
def dashjl_server() -> JuliaRunner:
    with JuliaRunner() as starter:
        yield starter