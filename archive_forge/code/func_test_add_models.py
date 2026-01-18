import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def test_add_models(self):
    num_new_models = 3
    num_new_topics = 3
    base_elda = self.get_elda()
    cumulative_elda = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_new_topics, passes=1, num_models=num_new_models, iterations=1, random_state=RANDOM_STATE, topic_model_class=LdaMulticore, workers=3, ensemble_workers=2)
    num_topics_before_add_model = len(cumulative_elda.ttda)
    num_models_before_add_model = cumulative_elda.num_models
    cumulative_elda.add_model(base_elda.ttda)
    assert len(cumulative_elda.ttda) == num_topics_before_add_model + len(base_elda.ttda)
    assert cumulative_elda.num_models == num_models_before_add_model + 1
    num_topics_before_add_model = len(cumulative_elda.ttda)
    num_models_before_add_model = cumulative_elda.num_models
    cumulative_elda.add_model(base_elda, 5)
    assert len(cumulative_elda.ttda) == num_topics_before_add_model + len(base_elda.ttda)
    assert cumulative_elda.num_models == num_models_before_add_model + 5
    num_topics_before_add_model = len(cumulative_elda.ttda)
    num_models_before_add_model = cumulative_elda.num_models
    base_elda_mem_unfriendly = self.get_elda_mem_unfriendly()
    cumulative_elda.add_model([base_elda, base_elda_mem_unfriendly])
    assert len(cumulative_elda.ttda) == num_topics_before_add_model + 2 * len(base_elda.ttda)
    assert cumulative_elda.num_models == num_models_before_add_model + 2 * NUM_MODELS
    model = base_elda.classic_model_representation
    num_topics_before_add_model = len(cumulative_elda.ttda)
    num_models_before_add_model = cumulative_elda.num_models
    cumulative_elda.add_model(model)
    assert len(cumulative_elda.ttda) == num_topics_before_add_model + len(model.get_topics())
    assert cumulative_elda.num_models == num_models_before_add_model + 1
    num_topics_before_add_model = len(cumulative_elda.ttda)
    num_models_before_add_model = cumulative_elda.num_models
    cumulative_elda.add_model([model, model])
    assert len(cumulative_elda.ttda) == num_topics_before_add_model + 2 * len(model.get_topics())
    assert cumulative_elda.num_models == num_models_before_add_model + 2
    self.assert_ttda_is_valid(cumulative_elda)
    elda_mem_unfriendly = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_new_topics, passes=1, num_models=num_new_models, iterations=1, random_state=RANDOM_STATE, topic_model_class=LdaMulticore, workers=3, ensemble_workers=2, memory_friendly_ttda=False)
    num_topics_before_add_model = len(elda_mem_unfriendly.tms)
    num_models_before_add_model = elda_mem_unfriendly.num_models
    elda_mem_unfriendly.add_model(base_elda_mem_unfriendly)
    assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model + NUM_MODELS
    assert elda_mem_unfriendly.num_models == num_models_before_add_model + NUM_MODELS
    num_topics_before_add_model = len(elda_mem_unfriendly.tms)
    num_models_before_add_model = elda_mem_unfriendly.num_models
    elda_mem_unfriendly.add_model([base_elda_mem_unfriendly, base_elda_mem_unfriendly])
    assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model + 2 * NUM_MODELS
    assert elda_mem_unfriendly.num_models == num_models_before_add_model + 2 * NUM_MODELS
    num_topics_before_add_model = len(elda_mem_unfriendly.tms)
    num_models_before_add_model = elda_mem_unfriendly.num_models
    elda_mem_unfriendly.add_model(base_elda_mem_unfriendly.tms[0])
    assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model + 1
    assert elda_mem_unfriendly.num_models == num_models_before_add_model + 1
    num_topics_before_add_model = len(elda_mem_unfriendly.tms)
    num_models_before_add_model = elda_mem_unfriendly.num_models
    elda_mem_unfriendly.add_model(base_elda_mem_unfriendly.tms)
    assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model + NUM_MODELS
    assert elda_mem_unfriendly.num_models == num_models_before_add_model + NUM_MODELS
    num_topics_before_add_model = len(elda_mem_unfriendly.tms)
    num_models_before_add_model = elda_mem_unfriendly.num_models
    with pytest.raises(ValueError):
        elda_mem_unfriendly.add_model(base_elda_mem_unfriendly.tms[0].get_topics())
    assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model
    assert elda_mem_unfriendly.num_models == num_models_before_add_model
    assert elda_mem_unfriendly.num_models == len(elda_mem_unfriendly.tms)
    self.assert_ttda_is_valid(elda_mem_unfriendly)