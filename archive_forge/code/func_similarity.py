import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
def similarity(self, w1, w2):
    """Compute similarity based on Poincare distance between vectors for nodes `w1` and `w2`.

        Parameters
        ----------
        w1 : {str, int}
            Key for first node.
        w2 : {str, int}
            Key for second node.

        Returns
        -------
        float
            Similarity between the between the vectors for nodes `w1` and `w2` (between 0 and 1).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # What is the similarity between the words 'mammal' and 'carnivore'?
            >>> model.kv.similarity('mammal.n.01', 'carnivore.n.01')
            0.25162107631176484

        Raises
        ------
        KeyError
            If either of `w1` and `w2` is absent from vocab.

        """
    return 1 / (1 + self.distance(w1, w2))