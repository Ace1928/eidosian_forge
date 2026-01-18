import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def average_networks(in_files, ntwk_res_file, group_id):
    """
    Sums the edges of input networks and divides by the number of networks
    Writes the average network as .pck and .gexf and returns the name of the written networks
    """
    import networkx as nx
    import os.path as op
    import scipy.io as sio
    iflogger.info('Creating average network for group: %s', group_id)
    matlab_network_list = []
    if len(in_files) == 1:
        avg_ntwk = read_unknown_ntwk(in_files[0])
    else:
        count_to_keep_edge = np.round(len(in_files) / 2.0)
        iflogger.info('Number of networks: %i, an edge must occur in at least %i to remain in the average network', len(in_files), count_to_keep_edge)
        ntwk_res_file = read_unknown_ntwk(ntwk_res_file)
        iflogger.info('%i nodes found in network resolution file', ntwk_res_file.number_of_nodes())
        ntwk = remove_all_edges(ntwk_res_file)
        counting_ntwk = ntwk.copy()
        for index, subject in enumerate(in_files):
            tmp = _read_pickle(subject)
            iflogger.info('File %s has %i edges', subject, tmp.number_of_edges())
            edges = list(tmp.edges())
            for edge in edges:
                data = {}
                data = tmp.edge[edge[0]][edge[1]]
                data['count'] = 1
                if ntwk.has_edge(edge[0], edge[1]):
                    current = {}
                    current = ntwk.edge[edge[0]][edge[1]]
                    data = add_dicts_by_key(current, data)
                ntwk.add_edge(edge[0], edge[1], **data)
            nodes = list(tmp.nodes())
            for node in nodes:
                data = {}
                data = ntwk.nodes[node]
                if 'value' in tmp.nodes[node]:
                    data['value'] = data['value'] + tmp.nodes[node]['value']
                ntwk.add_node(node, **data)
        nodes = list(ntwk.nodes())
        edges = list(ntwk.edges())
        iflogger.info('Total network has %i edges', ntwk.number_of_edges())
        avg_ntwk = nx.Graph()
        newdata = {}
        for node in nodes:
            data = ntwk.nodes[node]
            newdata = data
            if 'value' in data:
                newdata['value'] = data['value'] / len(in_files)
                ntwk.nodes[node]['value'] = newdata
            avg_ntwk.add_node(node, **newdata)
        edge_dict = {}
        edge_dict['count'] = np.zeros((avg_ntwk.number_of_nodes(), avg_ntwk.number_of_nodes()))
        for edge in edges:
            data = ntwk.edge[edge[0]][edge[1]]
            if ntwk.edge[edge[0]][edge[1]]['count'] >= count_to_keep_edge:
                for key in list(data.keys()):
                    if not key == 'count':
                        data[key] = data[key] / len(in_files)
                ntwk.edge[edge[0]][edge[1]] = data
                avg_ntwk.add_edge(edge[0], edge[1], **data)
            edge_dict['count'][edge[0] - 1][edge[1] - 1] = ntwk.edge[edge[0]][edge[1]]['count']
        iflogger.info('After thresholding, the average network has %i edges', avg_ntwk.number_of_edges())
        avg_edges = avg_ntwk.edges()
        for edge in avg_edges:
            data = avg_ntwk.edge[edge[0]][edge[1]]
            for key in list(data.keys()):
                if not key == 'count':
                    edge_dict[key] = np.zeros((avg_ntwk.number_of_nodes(), avg_ntwk.number_of_nodes()))
                    edge_dict[key][edge[0] - 1][edge[1] - 1] = data[key]
        for key in list(edge_dict.keys()):
            tmp = {}
            network_name = group_id + '_' + key + '_average.mat'
            matlab_network_list.append(op.abspath(network_name))
            tmp[key] = edge_dict[key]
            sio.savemat(op.abspath(network_name), tmp)
            iflogger.info('Saving average network for key: %s as %s', key, op.abspath(network_name))
    network_name = group_id + '_average.pck'
    with open(op.abspath(network_name), 'wb') as f:
        pickle.dump(avg_ntwk, f, pickle.HIGHEST_PROTOCOL)
    iflogger.info('Saving average network as %s', op.abspath(network_name))
    avg_ntwk = fix_keys_for_gexf(avg_ntwk)
    network_name = group_id + '_average.gexf'
    nx.write_gexf(avg_ntwk, op.abspath(network_name))
    iflogger.info('Saving average network as %s', op.abspath(network_name))
    return (network_name, matlab_network_list)