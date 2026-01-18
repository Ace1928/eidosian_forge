import networkx as nx
from networkx.utils import py_random_state
@nx._dispatch(graphs=None)
def is_valid_joint_degree(joint_degrees):
    """Checks whether the given joint degree dictionary is realizable.

    A *joint degree dictionary* is a dictionary of dictionaries, in
    which entry ``joint_degrees[k][l]`` is an integer representing the
    number of edges joining nodes of degree *k* with nodes of degree
    *l*. Such a dictionary is realizable as a simple graph if and only
    if the following conditions are satisfied.

    - each entry must be an integer,
    - the total number of nodes of degree *k*, computed by
      ``sum(joint_degrees[k].values()) / k``, must be an integer,
    - the total number of edges joining nodes of degree *k* with
      nodes of degree *l* cannot exceed the total number of possible edges,
    - each diagonal entry ``joint_degrees[k][k]`` must be even (this is
      a convention assumed by the :func:`joint_degree_graph` function).


    Parameters
    ----------
    joint_degrees :  dictionary of dictionary of integers
        A joint degree dictionary in which entry ``joint_degrees[k][l]``
        is the number of edges joining nodes of degree *k* with nodes of
        degree *l*.

    Returns
    -------
    bool
        Whether the given joint degree dictionary is realizable as a
        simple graph.

    References
    ----------
    .. [1] M. Gjoka, M. Kurant, A. Markopoulou, "2.5K Graphs: from Sampling
       to Generation", IEEE Infocom, 2013.
    .. [2] I. Stanton, A. Pinar, "Constructing and sampling graphs with a
       prescribed joint degree distribution", Journal of Experimental
       Algorithmics, 2012.
    """
    degree_count = {}
    for k in joint_degrees:
        if k > 0:
            k_size = sum(joint_degrees[k].values()) / k
            if not k_size.is_integer():
                return False
            degree_count[k] = k_size
    for k in joint_degrees:
        for l in joint_degrees[k]:
            if not float(joint_degrees[k][l]).is_integer():
                return False
            if k != l and joint_degrees[k][l] > degree_count[k] * degree_count[l]:
                return False
            elif k == l:
                if joint_degrees[k][k] > degree_count[k] * (degree_count[k] - 1):
                    return False
                if joint_degrees[k][k] % 2 != 0:
                    return False
    return True