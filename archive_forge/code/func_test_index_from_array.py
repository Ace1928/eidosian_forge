import pandas as pd
def test_index_from_array(self, data):
    idx = pd.Index(data)
    assert data.dtype == idx.dtype