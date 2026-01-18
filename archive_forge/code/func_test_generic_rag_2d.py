import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
def test_generic_rag_2d():
    labels = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    g = graph.RAG(labels)
    assert g.has_edge(1, 2) and g.has_edge(2, 4) and (not g.has_edge(1, 4))
    h = graph.RAG(labels, connectivity=2)
    assert h.has_edge(1, 2) and h.has_edge(1, 4) and h.has_edge(2, 3)