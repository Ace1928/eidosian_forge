import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
def test_generic_rag_3d():
    labels = np.arange(8, dtype=np.uint8).reshape((2, 2, 2))
    g = graph.RAG(labels)
    assert g.has_edge(0, 1) and g.has_edge(1, 3) and (not g.has_edge(0, 3))
    h = graph.RAG(labels, connectivity=2)
    assert h.has_edge(0, 1) and h.has_edge(0, 3) and (not h.has_edge(0, 7))
    k = graph.RAG(labels, connectivity=3)
    assert k.has_edge(0, 1) and k.has_edge(1, 2) and k.has_edge(2, 5)