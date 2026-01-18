import logging
from itertools import chain
from copy import deepcopy
from shutil import copyfile
from os.path import isfile
from os import remove
import numpy as np  # for arrays, array broadcasting etc.
from scipy.special import gammaln  # gamma function utils
from gensim import utils
from gensim.models import LdaModel
from gensim.models.ldamodel import LdaState
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.corpora import MmCorpus
def get_author_topics(self, author_name, minimum_probability=None):
    """Get topic distribution the given author.

        Parameters
        ----------
        author_name : str
            Name of the author for which the topic distribution needs to be estimated.
        minimum_probability : float, optional
            Sets the minimum probability value for showing the topics of a given author, topics with probability <
            `minimum_probability` will be ignored.

        Returns
        -------
        list of (int, float)
            Topic distribution of an author.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.models import AuthorTopicModel
            >>> from gensim.corpora import mmcorpus
            >>> from gensim.test.utils import common_dictionary, datapath, temporary_file

            >>> author2doc = {
            ...     'john': [0, 1, 2, 3, 4, 5, 6],
            ...     'jane': [2, 3, 4, 5, 6, 7, 8],
            ...     'jack': [0, 2, 4, 6, 8]
            ... }
            >>>
            >>> corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
            >>>
            >>> with temporary_file("serialized") as s_path:
            ...     model = AuthorTopicModel(
            ...         corpus, author2doc=author2doc, id2word=common_dictionary, num_topics=4,
            ...         serialized=True, serialization_path=s_path
            ...     )
            ...
            ...     model.update(corpus, author2doc)  # update the author-topic model with additional documents
            >>>
            >>> # construct vectors for authors
            >>> author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]

        """
    author_id = self.author2id[author_name]
    if minimum_probability is None:
        minimum_probability = self.minimum_probability
    minimum_probability = max(minimum_probability, 1e-08)
    topic_dist = self.state.gamma[author_id, :] / sum(self.state.gamma[author_id, :])
    author_topics = [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist) if topicvalue >= minimum_probability]
    return author_topics