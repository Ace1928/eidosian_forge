from ._internal import NDArrayBase
from ..base import _Null
def dgl_csr_neighbor_non_uniform_sample(*seed_arrays, **kwargs):
    """This operator samples sub-graph from a csr graph via an
    non-uniform probability. The operator is designed for DGL.

    The operator outputs four sets of NDArrays to represent the sampled results
    (the number of NDArrays in each set is the same as the number of seed NDArrays minus two (csr matrix and probability)):
    1) a set of 1D NDArrays containing the sampled vertices, 2) a set of CSRNDArrays representing
    the sampled edges, 3) a set of 1D NDArrays with the probability that vertices are sampled,
    4) a set of 1D NDArrays indicating the layer where a vertex is sampled.
    The first set of 1D NDArrays have a length of max_num_vertices+1. The last element in an NDArray
    indicate the acutal number of vertices in a subgraph. The third and fourth set of NDArrays have a length
    of max_num_vertices, and the valid number of vertices is the same as the ones in the first set.

    Example:

       .. code:: python

      shape = (5, 5)
      prob = mx.nd.array([0.9, 0.8, 0.2, 0.4, 0.1], dtype=np.float32)
      data_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=np.int64)
      indices_np = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3], dtype=np.int64)
      indptr_np = np.array([0,4,8,12,16,20], dtype=np.int64)
      a = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
      seed = mx.nd.array([0,1,2,3,4], dtype=np.int64)
      out = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(a, prob, seed, num_args=3, num_hops=1, num_neighbor=2, max_num_vertices=5)

      out[0]
      [0 1 2 3 4 5]
      <NDArray 6 @cpu(0)>

      out[1].asnumpy()
      array([[ 0,  1,  2,  0,  0],
             [ 5,  0,  6,  0,  0],
             [ 9, 10,  0,  0,  0],
             [13, 14,  0,  0,  0],
             [ 0, 18, 19,  0,  0]])

      out[2]
      [0.9 0.8 0.2 0.4 0.1]
      <NDArray 5 @cpu(0)>

      out[3]
      [0 0 0 0 0]
      <NDArray 5 @cpu(0)>



    Defined in ../src/operator/contrib/dgl_graph.cc:L911

    Parameters
    ----------
    csr_matrix : NDArray
        csr matrix
    probability : NDArray
        probability vector
    seed_arrays : NDArray[]
        seed vertices
    num_hops : long, optional, default=1
        Number of hops.
    num_neighbor : long, optional, default=2
        Number of neighbor.
    max_num_vertices : long, optional, default=100
        Max number of vertices.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)