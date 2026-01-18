from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout
@pytest.fixture
def nodes_without_positions():
    nodes_df = pd.DataFrame({'id': np.arange(5)})
    return nodes_df.set_index('id')