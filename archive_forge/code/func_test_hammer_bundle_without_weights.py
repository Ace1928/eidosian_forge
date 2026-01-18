from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout
@pytest.mark.parametrize('include_edge_id', [True, False])
def test_hammer_bundle_without_weights(nodes, edges, include_edge_id):
    data = pd.DataFrame({'edge_id': [1.0, np.nan, 2.0, np.nan, 3.0, np.nan, 4.0, np.nan], 'x': [0.0, np.nan, 0.0, np.nan, 0.0, np.nan, 0.0, np.nan], 'y': [0.0, np.nan, 0.0, np.nan, 0.0, np.nan, 0.0, np.nan]})
    columns = ['edge_id', 'x', 'y'] if include_edge_id else ['x', 'y']
    expected = pd.DataFrame(data, columns=columns)
    df = hammer_bundle(nodes, edges, include_edge_id=include_edge_id)
    starts = df[(df.x == 0.0) & (df.y == 0.0)]
    ends = df[df.isnull().any(axis=1)]
    given = pd.concat([starts, ends])
    given.sort_index(inplace=True)
    given.reset_index(drop=True, inplace=True)
    assert given.equals(expected)