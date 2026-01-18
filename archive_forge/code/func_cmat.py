import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
def cmat(track_file, roi_file, resolution_network_file, matrix_name, matrix_mat_name, endpoint_name, intersections=False):
    """Create the connection matrix for each resolution using fibers and ROIs."""
    import scipy.io as sio
    stats = {}
    iflogger.info('Running cmat function')
    en_fname = op.abspath(endpoint_name + '_endpoints.npy')
    en_fnamemm = op.abspath(endpoint_name + '_endpointsmm.npy')
    iflogger.info('Reading Trackvis file %s', track_file)
    fib, hdr = nb.trackvis.read(track_file, False)
    stats['orig_n_fib'] = len(fib)
    roi = nb.load(roi_file)
    roiData = np.asanyarray(roi.dataobj)
    roiVoxelSize = roi.header.get_zooms()
    endpoints, endpointsmm = create_endpoints_array(fib, roiVoxelSize)
    iflogger.info('Saving endpoint array: %s', en_fname)
    np.save(en_fname, endpoints)
    iflogger.info('Saving endpoint array in mm: %s', en_fnamemm)
    np.save(en_fnamemm, endpointsmm)
    n = len(fib)
    iflogger.info('Number of fibers: %i', n)
    fiberlabels = np.zeros((n, 2))
    final_fiberlabels = []
    final_fibers_idx = []
    path, name, ext = split_filename(resolution_network_file)
    if ext == '.pck':
        with open(resolution_network_file, 'rb') as f:
            gp = pickle.load(f)
    elif ext == '.graphml':
        gp = nx.read_graphml(resolution_network_file)
    else:
        raise TypeError('Unable to read file:', resolution_network_file)
    nROIs = len(gp.nodes())
    if 'dn_position' in gp.nodes[list(gp.nodes())[0]]:
        G = gp.copy()
    else:
        G = nx.Graph()
        for u, d in gp.nodes(data=True):
            G.add_node(int(u), **d)
            xyz = tuple(np.mean(np.where(np.flipud(roiData) == int(d['dn_correspondence_id'])), axis=1))
            G.nodes[int(u)]['dn_position'] = tuple([xyz[0], xyz[2], -xyz[1]])
    if intersections:
        iflogger.info('Filtering tractography from intersections')
        intersection_matrix, final_fiber_ids = create_allpoints_cmat(fib, roiData, roiVoxelSize, nROIs)
        finalfibers_fname = op.abspath(endpoint_name + '_intersections_streamline_final.trk')
        stats['intersections_n_fib'] = save_fibers(hdr, fib, finalfibers_fname, final_fiber_ids)
        intersection_matrix = np.matrix(intersection_matrix)
        I = G.copy()
        H = nx.from_numpy_array(np.matrix(intersection_matrix))
        H = nx.relabel_nodes(H, lambda x: x + 1)
        I.add_weighted_edges_from(((u, v, d['weight']) for u, v, d in H.edges(data=True)))
    dis = 0
    for i in range(endpoints.shape[0]):
        try:
            startROI = int(roiData[endpoints[i, 0, 0], endpoints[i, 0, 1], endpoints[i, 0, 2]])
            endROI = int(roiData[endpoints[i, 1, 0], endpoints[i, 1, 1], endpoints[i, 1, 2]])
        except IndexError:
            iflogger.error('AN INDEXERROR EXCEPTION OCCURRED FOR FIBER %s. PLEASE CHECK ENDPOINT GENERATION', i)
            break
        if startROI == 0 or endROI == 0:
            dis += 1
            fiberlabels[i, 0] = -1
            continue
        if startROI > nROIs or endROI > nROIs:
            iflogger.error('Start or endpoint of fiber terminate in a voxel which is labeled higher')
            iflogger.error('than is expected by the parcellation node information.')
            iflogger.error('Start ROI: %i, End ROI: %i', startROI, endROI)
            iflogger.error('This needs bugfixing!')
            continue
        if endROI < startROI:
            tmp = startROI
            startROI = endROI
            endROI = tmp
        fiberlabels[i, 0] = startROI
        fiberlabels[i, 1] = endROI
        final_fiberlabels.append([startROI, endROI])
        final_fibers_idx.append(i)
        if G.has_edge(startROI, endROI) and 'fiblist' in G.edge[startROI][endROI]:
            G.edge[startROI][endROI]['fiblist'].append(i)
        else:
            G.add_edge(startROI, endROI, fiblist=[i])
    finalfiberlength = []
    if intersections:
        final_fibers_indices = final_fiber_ids
    else:
        final_fibers_indices = final_fibers_idx
    for idx in final_fibers_indices:
        finalfiberlength.append(length(fib[idx][0]))
    final_fiberlength_array = np.array(finalfiberlength)
    final_fiberlabels_array = np.array(final_fiberlabels, dtype=int)
    iflogger.info('Found %i (%f percent out of %i fibers) fibers that start or terminate in a voxel which is not labeled. (orphans)', dis, dis * 100.0 / n, n)
    iflogger.info('Valid fibers: %i (%f%%)', n - dis, 100 - dis * 100.0 / n)
    numfib = nx.Graph()
    numfib.add_nodes_from(G)
    fibmean = numfib.copy()
    fibmedian = numfib.copy()
    fibdev = numfib.copy()
    for u, v, d in G.edges(data=True):
        G.remove_edge(u, v)
        di = {}
        if 'fiblist' in d:
            di['number_of_fibers'] = len(d['fiblist'])
            idx = np.where((final_fiberlabels_array[:, 0] == int(u)) & (final_fiberlabels_array[:, 1] == int(v)))[0]
            di['fiber_length_mean'] = float(np.mean(final_fiberlength_array[idx]))
            di['fiber_length_median'] = float(np.median(final_fiberlength_array[idx]))
            di['fiber_length_std'] = float(np.std(final_fiberlength_array[idx]))
        else:
            di['number_of_fibers'] = 0
            di['fiber_length_mean'] = 0
            di['fiber_length_median'] = 0
            di['fiber_length_std'] = 0
        if not u == v:
            G.add_edge(u, v, **di)
            if 'fiblist' in d:
                numfib.add_edge(u, v, weight=di['number_of_fibers'])
                fibmean.add_edge(u, v, weight=di['fiber_length_mean'])
                fibmedian.add_edge(u, v, weight=di['fiber_length_median'])
                fibdev.add_edge(u, v, weight=di['fiber_length_std'])
    iflogger.info('Writing network as %s', matrix_name)
    with open(op.abspath(matrix_name), 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    numfib_mlab = nx.to_numpy_array(numfib, dtype=int)
    numfib_dict = {'number_of_fibers': numfib_mlab}
    fibmean_mlab = nx.to_numpy_array(fibmean, dtype=np.float64)
    fibmean_dict = {'mean_fiber_length': fibmean_mlab}
    fibmedian_mlab = nx.to_numpy_array(fibmedian, dtype=np.float64)
    fibmedian_dict = {'median_fiber_length': fibmedian_mlab}
    fibdev_mlab = nx.to_numpy_array(fibdev, dtype=np.float64)
    fibdev_dict = {'fiber_length_std': fibdev_mlab}
    if intersections:
        path, name, ext = split_filename(matrix_name)
        intersection_matrix_name = op.abspath(name + '_intersections') + ext
        iflogger.info('Writing intersection network as %s', intersection_matrix_name)
        with open(intersection_matrix_name, 'wb') as f:
            pickle.dump(I, f, pickle.HIGHEST_PROTOCOL)
    path, name, ext = split_filename(matrix_mat_name)
    if not ext == '.mat':
        ext = '.mat'
        matrix_mat_name = matrix_mat_name + ext
    iflogger.info('Writing matlab matrix as %s', matrix_mat_name)
    sio.savemat(matrix_mat_name, numfib_dict)
    if intersections:
        intersect_dict = {'intersections': intersection_matrix}
        intersection_matrix_mat_name = op.abspath(name + '_intersections') + ext
        iflogger.info('Writing intersection matrix as %s', intersection_matrix_mat_name)
        sio.savemat(intersection_matrix_mat_name, intersect_dict)
    mean_fiber_length_matrix_name = op.abspath(name + '_mean_fiber_length') + ext
    iflogger.info('Writing matlab mean fiber length matrix as %s', mean_fiber_length_matrix_name)
    sio.savemat(mean_fiber_length_matrix_name, fibmean_dict)
    median_fiber_length_matrix_name = op.abspath(name + '_median_fiber_length') + ext
    iflogger.info('Writing matlab median fiber length matrix as %s', median_fiber_length_matrix_name)
    sio.savemat(median_fiber_length_matrix_name, fibmedian_dict)
    fiber_length_std_matrix_name = op.abspath(name + '_fiber_length_std') + ext
    iflogger.info('Writing matlab fiber length deviation matrix as %s', fiber_length_std_matrix_name)
    sio.savemat(fiber_length_std_matrix_name, fibdev_dict)
    fiberlengths_fname = op.abspath(endpoint_name + '_final_fiberslength.npy')
    iflogger.info('Storing final fiber length array as %s', fiberlengths_fname)
    np.save(fiberlengths_fname, final_fiberlength_array)
    fiberlabels_fname = op.abspath(endpoint_name + '_filtered_fiberslabel.npy')
    iflogger.info('Storing all fiber labels (with orphans) as %s', fiberlabels_fname)
    np.save(fiberlabels_fname, np.array(fiberlabels, dtype=np.int32))
    fiberlabels_noorphans_fname = op.abspath(endpoint_name + '_final_fiberslabels.npy')
    iflogger.info('Storing final fiber labels (no orphans) as %s', fiberlabels_noorphans_fname)
    np.save(fiberlabels_noorphans_fname, final_fiberlabels_array)
    iflogger.info('Filtering tractography - keeping only no orphan fibers')
    finalfibers_fname = op.abspath(endpoint_name + '_streamline_final.trk')
    stats['endpoint_n_fib'] = save_fibers(hdr, fib, finalfibers_fname, final_fibers_idx)
    stats['endpoints_percent'] = float(stats['endpoint_n_fib']) / float(stats['orig_n_fib']) * 100
    stats['intersections_percent'] = float(stats['intersections_n_fib']) / float(stats['orig_n_fib']) * 100
    out_stats_file = op.abspath(endpoint_name + '_statistics.mat')
    iflogger.info('Saving matrix creation statistics as %s', out_stats_file)
    sio.savemat(out_stats_file, stats)