import numpy as np
def n_0(m_array, coloured):
    m_coloured = m_array[list(coloured)]
    l_val = m_coloured[-1]
    for i in range(len(m_coloured) - 1):
        l_val += m_coloured[i]
    white_neighbours = np.argwhere(np.logical_not(l_val))
    return {x[0] for x in white_neighbours} - coloured