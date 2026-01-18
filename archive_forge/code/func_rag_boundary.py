import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
def rag_boundary(labels, edge_map, connectivity=2):
    """Comouter RAG based on region boundaries

    Given an image's initial segmentation and its edge map this method
    constructs the corresponding Region Adjacency Graph (RAG). Each node in the
    RAG represents a set of pixels within the image with the same label in
    `labels`. The weight between two adjacent regions is the average value
    in `edge_map` along their boundary.

    labels : ndarray
        The labelled image.
    edge_map : ndarray
        This should have the same shape as that of `labels`. For all pixels
        along the boundary between 2 adjacent regions, the average value of the
        corresponding pixels in `edge_map` is the edge weight between them.
    connectivity : int, optional
        Pixels with a squared distance less than `connectivity` from each other
        are considered adjacent. It can range from 1 to `labels.ndim`. Its
        behavior is the same as `connectivity` parameter in
        `scipy.ndimage.generate_binary_structure`.

    Examples
    --------
    >>> from skimage import data, segmentation, filters, color, graph
    >>> img = data.chelsea()
    >>> labels = segmentation.slic(img)
    >>> edge_map = filters.sobel(color.rgb2gray(img))
    >>> rag = graph.rag_boundary(labels, edge_map)

    """
    conn = ndi.generate_binary_structure(labels.ndim, connectivity)
    eroded = ndi.grey_erosion(labels, footprint=conn)
    dilated = ndi.grey_dilation(labels, footprint=conn)
    boundaries0 = eroded != labels
    boundaries1 = dilated != labels
    labels_small = np.concatenate((eroded[boundaries0], labels[boundaries1]))
    labels_large = np.concatenate((labels[boundaries0], dilated[boundaries1]))
    n = np.max(labels_large) + 1
    ones = np.broadcast_to(1.0, labels_small.shape)
    count_matrix = sparse.coo_matrix((ones, (labels_small, labels_large)), dtype=int, shape=(n, n)).tocsr()
    data = np.concatenate((edge_map[boundaries0], edge_map[boundaries1]))
    data_coo = sparse.coo_matrix((data, (labels_small, labels_large)))
    graph_matrix = data_coo.tocsr()
    graph_matrix.data /= count_matrix.data
    rag = RAG()
    rag.add_weighted_edges_from(_edge_generator_from_csr(graph_matrix), weight='weight')
    rag.add_weighted_edges_from(_edge_generator_from_csr(count_matrix), weight='count')
    for n in rag.nodes():
        rag.nodes[n].update({'labels': [n]})
    return rag