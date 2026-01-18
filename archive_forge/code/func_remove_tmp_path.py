import shutil
import pytest
from tensorflow import keras
@pytest.fixture(autouse=True)
def remove_tmp_path(tmp_path):
    yield
    shutil.rmtree(tmp_path)