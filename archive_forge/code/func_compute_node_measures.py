import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def compute_node_measures(ntwk, calculate_cliques=False):
    """
    These return node-based measures
    """
    iflogger.info('Computing node measures:')
    measures = {}
    iflogger.info('...Computing degree...')
    measures['degree'] = np.array(list(ntwk.degree().values()))
    iflogger.info('...Computing load centrality...')
    measures['load_centrality'] = np.array(list(nx.load_centrality(ntwk).values()))
    iflogger.info('...Computing betweenness centrality...')
    measures['betweenness_centrality'] = np.array(list(nx.betweenness_centrality(ntwk).values()))
    iflogger.info('...Computing degree centrality...')
    measures['degree_centrality'] = np.array(list(nx.degree_centrality(ntwk).values()))
    iflogger.info('...Computing closeness centrality...')
    measures['closeness_centrality'] = np.array(list(nx.closeness_centrality(ntwk).values()))
    iflogger.info('...Computing triangles...')
    measures['triangles'] = np.array(list(nx.triangles(ntwk).values()))
    iflogger.info('...Computing clustering...')
    measures['clustering'] = np.array(list(nx.clustering(ntwk).values()))
    iflogger.info('...Computing k-core number')
    measures['core_number'] = np.array(list(nx.core_number(ntwk).values()))
    iflogger.info('...Identifying network isolates...')
    isolate_list = nx.isolates(ntwk)
    binarized = np.zeros((ntwk.number_of_nodes(), 1))
    for value in isolate_list:
        value = value - 1
        binarized[value] = 1
    measures['isolates'] = binarized
    if calculate_cliques:
        iflogger.info('...Calculating node clique number')
        measures['node_clique_number'] = np.array(list(nx.node_clique_number(ntwk).values()))
        iflogger.info('...Computing number of cliques for each node...')
        measures['number_of_cliques'] = np.array(list(nx.number_of_cliques(ntwk).values()))
    return measures