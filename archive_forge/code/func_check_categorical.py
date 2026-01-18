import pytest
@pytest.fixture(params=[True, False])
def check_categorical(request):
    return request.param