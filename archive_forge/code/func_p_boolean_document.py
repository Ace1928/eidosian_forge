import itertools
import logging
from gensim.topic_coherence.text_analysis import (
def p_boolean_document(corpus, segmented_topics):
    """Perform the boolean document probability estimation. Boolean document estimates the probability of a single word
    as the number of documents in which the word occurs divided by the total number of documents.

    Parameters
    ----------
    corpus : iterable of list of (int, int)
        The corpus of documents.
    segmented_topics: list of (int, int).
        Each tuple (word_id_set1, word_id_set2) is either a single integer, or a `numpy.ndarray` of integers.

    Returns
    -------
    :class:`~gensim.topic_coherence.text_analysis.CorpusAccumulator`
        Word occurrence accumulator instance that can be used to lookup token frequencies and co-occurrence frequencies.

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
        >>>
        >>> result = probability_estimation.p_boolean_document(corpus, segmented_topics)
        >>> result.index_to_dict()
        {10608: set([0]), 12736: set([1, 3]), 18451: set([5]), 5798: set([1, 2])}

    """
    top_ids = unique_ids_from_segments(segmented_topics)
    return CorpusAccumulator(top_ids).accumulate(corpus)