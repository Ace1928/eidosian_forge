import pytest
from .consts import SELENIUM_GRID_DEFAULT
@pytest.fixture
def dash_multi_process_server() -> MultiProcessRunner:
    with MultiProcessRunner() as starter:
        yield starter