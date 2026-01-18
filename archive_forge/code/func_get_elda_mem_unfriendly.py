import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def get_elda_mem_unfriendly(self):
    return EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS, passes=PASSES, num_models=NUM_MODELS, random_state=RANDOM_STATE, memory_friendly_ttda=False, topic_model_class=LdaModel)