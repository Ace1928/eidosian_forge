import itertools
import numpy as np
import pandas as pd
import param
from ..core import Dataset
from ..core.boundingregion import BoundingBox
from ..core.data import PandasInterface, default_datatype
from ..core.operation import Operation
from ..core.sheetcoords import Slice
from ..core.util import (
def connect_tri_edges_pd(trimesh):
    """
    Given a TriMesh element containing abstract edges compute edge
    segments directly connecting the source and target nodes. This
    operation depends on pandas and is a lot faster than the pure
    NumPy equivalent.
    """
    edges = trimesh.dframe().copy()
    edges.index.name = 'trimesh_edge_index'
    edges = edges.drop('color', errors='ignore', axis=1).reset_index()
    nodes = trimesh.nodes.dframe().copy()
    nodes.index.name = 'node_index'
    nodes = nodes.drop(['color', 'z'], errors='ignore', axis=1)
    v1, v2, v3 = trimesh.kdims
    x, y, idx = trimesh.nodes.kdims[:3]
    df = pd.merge(edges, nodes, left_on=[v1.name], right_on=[idx.name])
    df = df.rename(columns={x.name: 'x0', y.name: 'y0'})
    df = pd.merge(df, nodes, left_on=[v2.name], right_on=[idx.name])
    df = df.rename(columns={x.name: 'x1', y.name: 'y1'})
    df = pd.merge(df, nodes, left_on=[v3.name], right_on=[idx.name])
    df = df.rename(columns={x.name: 'x2', y.name: 'y2'})
    df = df.sort_values('trimesh_edge_index').drop(['trimesh_edge_index'], axis=1)
    return df[['x0', 'y0', 'x1', 'y1', 'x2', 'y2']]