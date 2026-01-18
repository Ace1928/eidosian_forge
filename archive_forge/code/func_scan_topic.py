import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
def scan_topic(topic_index, current_label=None, parent_neighbors=None):
    """Extend the cluster in one direction.

            Results are accumulated to ``self.results``.

            Parameters
            ----------
            topic_index : int
                The topic that might be added to the existing cluster, or which might create a new cluster if necessary.
            current_label : int
                The label of the cluster that might be suitable for ``topic_index``

            """
    neighbors_sorted = sorted([(distance, index) for index, distance in enumerate(amatrix_copy[topic_index])], key=lambda x: x[0])
    neighboring_topic_indices = [index for distance, index in neighbors_sorted if distance < self.eps]
    num_neighboring_topics = len(neighboring_topic_indices)
    if num_neighboring_topics >= self.min_samples:
        topic_clustering_results[topic_index].is_core = True
        if current_label is None:
            current_label = self.next_label
            self.next_label += 1
        else:
            close_parent_neighbors_mask = amatrix_copy[topic_index][parent_neighbors] < self.eps
            if close_parent_neighbors_mask.mean() < 0.25:
                current_label = self.next_label
                self.next_label += 1
        topic_clustering_results[topic_index].label = current_label
        for neighboring_topic_index in neighboring_topic_indices:
            if topic_clustering_results[neighboring_topic_index].label is None:
                ordered_min_similarity.remove(neighboring_topic_index)
                scan_topic(neighboring_topic_index, current_label, neighboring_topic_indices + [topic_index])
            topic_clustering_results[neighboring_topic_index].neighboring_topic_indices.add(topic_index)
            topic_clustering_results[neighboring_topic_index].neighboring_labels.add(current_label)
    elif current_label is None:
        topic_clustering_results[topic_index].label = -1
    else:
        topic_clustering_results[topic_index].label = current_label