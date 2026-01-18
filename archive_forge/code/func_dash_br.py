import pytest
from .consts import SELENIUM_GRID_DEFAULT
@pytest.fixture
def dash_br(request, tmpdir) -> Browser:
    with Browser(browser=request.config.getoption('webdriver'), remote=request.config.getoption('remote'), remote_url=request.config.getoption('remote_url'), headless=request.config.getoption('headless'), options=request.config.hook.pytest_setup_options(), download_path=tmpdir.mkdir('download').strpath, percy_assets_root=request.config.getoption('percy_assets'), percy_finalize=request.config.getoption('nopercyfinalize'), pause=request.config.getoption('pause')) as browser:
        yield browser