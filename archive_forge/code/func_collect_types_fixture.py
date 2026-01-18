import pytest
@pytest.fixture(autouse=True)
def collect_types_fixture():
    from pyannotate_runtime import collect_types
    collect_types.resume()
    yield
    collect_types.pause()