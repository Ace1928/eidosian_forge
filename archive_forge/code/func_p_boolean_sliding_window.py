import itertools
import logging
from gensim.topic_coherence.text_analysis import (
def p_boolean_sliding_window(texts, segmented_topics, dictionary, window_size, processes=1):
    """Perform the boolean sliding window probability estimation.

    Parameters
    ----------
    texts : iterable of iterable of str
        Input text
    segmented_topics: list of (int, int)
        Each tuple (word_id_set1, word_id_set2) is either a single integer, or a `numpy.ndarray` of integers.
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        Gensim dictionary mapping of the tokens and ids.
    window_size : int
        Size of the sliding window, 110 found out to be the ideal size for large corpora.
    processes : int, optional
        Number of process that will be used for
        :class:`~gensim.topic_coherence.text_analysis.ParallelWordOccurrenceAccumulator`

    Notes
    -----
    Boolean sliding window determines word counts using a sliding window. The window
    moves over  the documents one word token per step. Each step defines a new virtual
    document  by copying the window content. Boolean document is applied to these virtual
    documents to compute word probabilities.

    Returns
    -------
    :class:`~gensim.topic_coherence.text_analysis.WordOccurrenceAccumulator`
        if `processes` = 1 OR
    :class:`~gensim.topic_coherence.text_analysis.ParallelWordOccurrenceAccumulator`
        otherwise. This is word occurrence accumulator instance that can be used to lookup
        token frequencies and co-occurrence frequencies.

    Examples
    ---------
    .. sourcecode:: pycon

        >>> from gensim.topic_coherence import probability_estimation
        >>> from gensim.corpora.hashdictionary import HashDictionary
        >>>
        >>>
        >>> texts = [
        ...     ['human', 'interface', 'computer'],
        ...     ['eps', 'user', 'interface', 'system'],
        ...     ['system', 'human', 'system', 'eps'],
        ...     ['user', 'response', 'time'],
        ...     ['trees'],
        ...     ['graph', 'trees']
        ... ]
        >>> dictionary = HashDictionary(texts)
        >>> w2id = dictionary.token2id

        >>>
        >>> # create segmented_topics
        >>> segmented_topics = [
        ...     [
        ...         (w2id['system'], w2id['graph']),
        ...         (w2id['computer'], w2id['graph']),
        ...         (w2id['computer'], w2id['system'])
        ...     ],
        ...     [
        ...         (w2id['computer'], w2id['graph']),
        ...         (w2id['user'], w2id['graph']),
        ...         (w2id['user'], w2id['computer'])]
        ... ]
        >>> # create corpus
        >>> corpus = [dictionary.doc2bow(text) for text in texts]
        >>> accumulator = probability_estimation.p_boolean_sliding_window(texts, segmented_topics, dictionary, 2)
        >>>
        >>> (accumulator[w2id['computer']], accumulator[w2id['user']], accumulator[w2id['system']])
        (1, 3, 4)

    """
    top_ids = unique_ids_from_segments(segmented_topics)
    if processes <= 1:
        accumulator = WordOccurrenceAccumulator(top_ids, dictionary)
    else:
        accumulator = ParallelWordOccurrenceAccumulator(processes, top_ids, dictionary)
    logger.info('using %s to estimate probabilities from sliding windows', accumulator)
    return accumulator.accumulate(texts, window_size)