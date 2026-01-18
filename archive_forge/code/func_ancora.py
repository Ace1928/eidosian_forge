import random
import pytest
from thinc.api import (
@pytest.fixture(scope='module')
def ancora():
    pytest.importorskip('ml_datasets')
    import ml_datasets
    return ml_datasets.ud_ancora_pos_tags()