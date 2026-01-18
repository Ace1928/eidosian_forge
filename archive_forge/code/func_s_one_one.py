import logging
def s_one_one(topics):
    """Perform segmentation on a list of topics.
    Segmentation is defined as
    :math:`s_{one} = {(W', W^{*}) | W' = {w_i}; W^{*} = {w_j}; w_{i}, w_{j} \\in W; i \\neq j}`.

    Parameters
    ----------
    topics : list of `numpy.ndarray`
        List of topics obtained from an algorithm such as LDA.

    Returns
    -------
    list of list of (int, int).
        :math:`(W', W^{*})` for all unique topic ids.

    Examples
    -------
    .. sourcecode:: pycon

        >>> import numpy as np
        >>> from gensim.topic_coherence import segmentation
        >>>
        >>> topics = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> segmentation.s_one_one(topics)
        [[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)], [(4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5)]]

    """
    s_one_one_res = []
    for top_words in topics:
        s_one_one_t = []
        for w_prime_index, w_prime in enumerate(top_words):
            for w_star_index, w_star in enumerate(top_words):
                if w_prime_index == w_star_index:
                    continue
                else:
                    s_one_one_t.append((w_prime, w_star))
        s_one_one_res.append(s_one_one_t)
    return s_one_one_res