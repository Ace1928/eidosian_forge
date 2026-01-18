from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
@classmethod
def for_models(cls, models, dictionary, topn=20, **kwargs):
    """Initialize a CoherenceModel with estimated probabilities for all of the given models.
        Use :meth:`~gensim.models.coherencemodel.CoherenceModel.for_topics` method.

        Parameters
        ----------
        models : list of :class:`~gensim.models.basemodel.BaseTopicModel`
            List of models to evaluate coherence of, each of it should implements
            :meth:`~gensim.models.basemodel.BaseTopicModel.get_topics` method.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            Gensim dictionary mapping of id word.
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic.
        kwargs : object
            Sequence of arguments, see :meth:`~gensim.models.coherencemodel.CoherenceModel.for_topics`.

        Return
        ------
        :class:`~gensim.models.coherencemodel.CoherenceModel`
            CoherenceModel with estimated probabilities for all of the given models.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import common_corpus, common_dictionary
            >>> from gensim.models.ldamodel import LdaModel
            >>> from gensim.models.coherencemodel import CoherenceModel
            >>>
            >>> m1 = LdaModel(common_corpus, 3, common_dictionary)
            >>> m2 = LdaModel(common_corpus, 5, common_dictionary)
            >>>
            >>> cm = CoherenceModel.for_models([m1, m2], common_dictionary, corpus=common_corpus, coherence='u_mass')
        """
    topics = [cls.top_topics_as_word_lists(model, dictionary, topn) for model in models]
    kwargs['dictionary'] = dictionary
    kwargs['topn'] = topn
    return cls.for_topics(topics, **kwargs)