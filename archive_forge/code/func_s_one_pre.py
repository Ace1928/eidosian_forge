import logging
def s_one_pre(topics):
    """Performs segmentation on a list of topics.

    Notes
    -----
    Segmentation is defined as
    :math:`s_{pre} = {(W', W^{*}) | W' = w_{i}; W^{*} = {w_j}; w_{i}, w_{j} \\in W; i > j}`.

    Parameters
    ----------
    topics : list of np.array
        list of topics obtained from an algorithm such as LDA.

    Returns
    -------
    list of list of (int, int)
        :math:`(W', W^{*})` for all unique topic ids.

    Examples
    --------
    .. sourcecode:: pycon

        >>> import numpy as np
        >>> from gensim.topic_coherence import segmentation
        >>>
        >>> topics = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> segmentation.s_one_pre(topics)
        [[(2, 1), (3, 1), (3, 2)], [(5, 4), (6, 4), (6, 5)]]

    """
    s_one_pre_res = []
    for top_words in topics:
        s_one_pre_t = []
        for w_prime_index, w_prime in enumerate(top_words[1:]):
            for w_star in top_words[:w_prime_index + 1]:
                s_one_pre_t.append((w_prime, w_star))
        s_one_pre_res.append(s_one_pre_t)
    return s_one_pre_res