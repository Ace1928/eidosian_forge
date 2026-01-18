from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout
def test_immutable_nodes(nodes, edges):
    original = nodes.copy()
    directly_connect_edges(nodes, edges)
    assert original.equals(nodes)