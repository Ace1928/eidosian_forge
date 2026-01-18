import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def test_add_and_recluster(self):
    num_new_models = 3
    num_new_topics = 3
    random_state = 1
    elda_1 = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_new_topics, passes=10, num_models=num_new_models, iterations=30, random_state=random_state, topic_model_class='lda', distance_workers=4)
    elda_mem_unfriendly_1 = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_new_topics, passes=10, num_models=num_new_models, iterations=30, random_state=random_state, topic_model_class=LdaModel, distance_workers=4, memory_friendly_ttda=False)
    elda_2 = self.get_elda()
    elda_mem_unfriendly_2 = self.get_elda_mem_unfriendly()
    assert elda_1.random_state != elda_2.random_state
    assert elda_mem_unfriendly_1.random_state != elda_mem_unfriendly_2.random_state
    np.testing.assert_allclose(elda_1.ttda, elda_mem_unfriendly_1.ttda, rtol=RTOL)
    np.testing.assert_allclose(elda_1.get_topics(), elda_mem_unfriendly_1.get_topics(), rtol=RTOL)
    elda_1.add_model(elda_2)
    elda_mem_unfriendly_1.add_model(elda_mem_unfriendly_2)
    np.testing.assert_allclose(elda_1.ttda, elda_mem_unfriendly_1.ttda, rtol=RTOL)
    assert len(elda_1.ttda) == len(elda_2.ttda) + num_new_models * num_new_topics
    assert len(elda_mem_unfriendly_1.ttda) == len(elda_mem_unfriendly_2.ttda) + num_new_models * num_new_topics
    assert len(elda_mem_unfriendly_1.tms) == NUM_MODELS + num_new_models
    self.assert_ttda_is_valid(elda_1)
    self.assert_ttda_is_valid(elda_mem_unfriendly_1)
    elda_1._generate_asymmetric_distance_matrix()
    elda_mem_unfriendly_1._generate_asymmetric_distance_matrix()
    np.testing.assert_allclose(elda_1.asymmetric_distance_matrix, elda_mem_unfriendly_1.asymmetric_distance_matrix)
    elda_1._generate_topic_clusters()
    elda_mem_unfriendly_1._generate_topic_clusters()
    clustering_results = elda_1.cluster_model.results
    mem_unfriendly_clustering_results = elda_mem_unfriendly_1.cluster_model.results
    self.assert_clustering_results_equal(clustering_results, mem_unfriendly_clustering_results)
    elda_1._generate_stable_topics()
    elda_mem_unfriendly_1._generate_stable_topics()
    np.testing.assert_allclose(elda_1.get_topics(), elda_mem_unfriendly_1.get_topics())
    elda_1.generate_gensim_representation()
    elda_mem_unfriendly_1.generate_gensim_representation()
    np.testing.assert_allclose(elda_1.get_topics(), elda_mem_unfriendly_1.get_topics(), rtol=RTOL)