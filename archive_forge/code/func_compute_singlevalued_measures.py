import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def compute_singlevalued_measures(ntwk, weighted=True, calculate_cliques=False):
    """
    Returns a single value per network
    """
    iflogger.info('Computing single valued measures:')
    measures = {}
    iflogger.info('...Computing degree assortativity (pearson number) ...')
    measures['degree_pearsonr'] = nx.degree_pearson_correlation_coefficient(ntwk)
    iflogger.info('...Computing degree assortativity...')
    measures['degree_assortativity'] = nx.degree_assortativity_coefficient(ntwk)
    iflogger.info('...Computing transitivity...')
    measures['transitivity'] = nx.transitivity(ntwk)
    iflogger.info('...Computing number of connected_components...')
    measures['number_connected_components'] = nx.number_connected_components(ntwk)
    iflogger.info('...Computing graph density...')
    measures['graph_density'] = nx.density(ntwk)
    iflogger.info('...Recording number of edges...')
    measures['number_of_edges'] = nx.number_of_edges(ntwk)
    iflogger.info('...Recording number of nodes...')
    measures['number_of_nodes'] = nx.number_of_nodes(ntwk)
    iflogger.info('...Computing average clustering...')
    measures['average_clustering'] = nx.average_clustering(ntwk)
    if nx.is_connected(ntwk):
        iflogger.info('...Calculating average shortest path length...')
        measures['average_shortest_path_length'] = nx.average_shortest_path_length(ntwk, weighted)
    else:
        iflogger.info('...Calculating average shortest path length...')
        measures['average_shortest_path_length'] = nx.average_shortest_path_length(nx.connected_component_subgraphs(ntwk)[0], weighted)
    if calculate_cliques:
        iflogger.info('...Computing graph clique number...')
        measures['graph_clique_number'] = nx.graph_clique_number(ntwk)
    return measures