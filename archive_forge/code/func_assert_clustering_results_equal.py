import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
def assert_clustering_results_equal(self, clustering_results_1, clustering_results_2):
    """Assert important attributes of the cluster results"""
    np.testing.assert_array_equal([element.label for element in clustering_results_1], [element.label for element in clustering_results_2])
    np.testing.assert_array_equal([element.is_core for element in clustering_results_1], [element.is_core for element in clustering_results_2])