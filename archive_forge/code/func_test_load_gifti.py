from nibabel import load
from .test_parse_gifti_fast import DATA_FILE3
def test_load_gifti():
    load(DATA_FILE3)