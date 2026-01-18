import numpy as np
def recursive_largest_first(binary_observables, adj):
    """Performs graph-colouring using the Recursive Largest Degree First heuristic. Often yields a
    lower chromatic number than Largest Degree First, but takes longer (runtime is cubic in number
    of vertices).

    Args:
        binary_observables (array[int]): the set of Pauli words represented by a column matrix of
            the Pauli words in binary vector represenation
        adj (array[int]): the adjacency matrix of the Pauli graph

    Returns:
        dict(int, list[array[int]]): keys correspond to colours (labelled by integers) and values
        are lists of Pauli words of the same colour in binary vector representation

    **Example**

    >>> binary_observables = np.array([[1., 1., 0.],
    ... [1., 0., 0.],
    ... [0., 0., 1.],
    ... [1., 0., 1.]])
    >>> adj = np.array([[0., 0., 1.],
    ... [0., 0., 1.],
    ... [1., 1., 0.]])
    >>> recursive_largest_first(binary_observables, adj)
    {1: [array([0., 0., 1.])], 2: [array([1., 1., 0.]), array([1., 0., 0.])]}
    """

    def n_0(m_array, coloured):
        m_coloured = m_array[list(coloured)]
        l_val = m_coloured[-1]
        for i in range(len(m_coloured) - 1):
            l_val += m_coloured[i]
        white_neighbours = np.argwhere(np.logical_not(l_val))
        return {x[0] for x in white_neighbours} - coloured
    n_terms = np.shape(adj)[0]
    terms = [binary_observables[i] for i in range(n_terms)]
    colours = {}
    c_vec = np.zeros(n_terms, dtype=int)
    uncoloured = set(np.arange(n_terms))
    coloured = set()
    k = 0
    while uncoloured:
        decode = np.array(list(uncoloured))
        k += 1
        m_array = adj[:, decode][decode, :]
        v_indices = np.argmax(m_array.sum(axis=1))
        coloured_sub = {v_indices}
        uncoloured_sub = set(np.arange(len(decode))) - {v_indices}
        n0_set = n_0(m_array, coloured_sub)
        n1_set = uncoloured_sub - n0_set
        while n0_set:
            m_uncoloured = m_array[:, list(n1_set)][list(n0_set), :]
            v_indices = list(n0_set)[np.argmax(m_uncoloured.sum(axis=1))]
            coloured_sub.add(v_indices)
            uncoloured_sub -= {v_indices}
            n0_set = n_0(m_array, coloured_sub)
            n1_set = uncoloured_sub - n0_set
        indices = decode[list(coloured_sub)]
        c_vec[indices] = k
        colours[k] = [terms[i] for i in indices]
        coloured |= set(indices)
        uncoloured = set(np.arange(n_terms)) - coloured
    return colours