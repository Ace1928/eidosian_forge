from __future__ import annotations
import itertools
from warnings import warn
import networkx as nx
import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import cite_conventional_cell_algo
from pymatgen.symmetry.kpath import KPathBase, KPathLatimerMunro, KPathSeek, KPathSetyawanCurtarolo
@staticmethod
def get_continuous_path(bandstructure):
    """Obtain a continuous version of an inputted path using graph theory.
        This routine will attempt to add connections between nodes of
        odd-degree to ensure a Eulerian path can be formed. Initial
        k-path must be able to be converted to a connected graph. See
        npj Comput Mater 6, 112 (2020). 10.1038/s41524-020-00383-7
        for more details.

        Args:
            bandstructure (BandstructureSymmLine): BandstructureSymmLine object.

        Returns:
            bandstructure (BandstructureSymmLine): New BandstructureSymmLine object with continuous path.
        """
    G = nx.Graph()
    labels = [point.label for point in bandstructure.kpoints if point.label is not None]
    plot_axis = []
    for i in range(int(len(labels) / 2)):
        G.add_edges_from([(labels[2 * i], labels[2 * i + 1])])
        plot_axis.append((labels[2 * i], labels[2 * i + 1]))
    G_euler = nx.algorithms.euler.eulerize(G)
    G_euler_circuit = nx.algorithms.euler.eulerian_circuit(G_euler)
    distances_map = []
    kpath_euler = []
    for edge_euler in G_euler_circuit:
        kpath_euler.append(edge_euler)
        for edge_reg in plot_axis:
            if edge_euler == edge_reg:
                distances_map.append((plot_axis.index(edge_reg), False))
            elif edge_euler[::-1] == edge_reg:
                distances_map.append((plot_axis.index(edge_reg), True))
    spins = [Spin.up, Spin.down] if bandstructure.is_spin_polarized else [Spin.up]
    new_kpoints = []
    new_bands = {spin: [np.array([]) for _ in range(bandstructure.nb_bands)] for spin in spins}
    new_projections = {spin: [[] for _ in range(bandstructure.nb_bands)] for spin in spins}
    n_branches = len(bandstructure.branches)
    new_branches = []
    processed = []
    for idx in range(n_branches):
        branch = bandstructure.branches[idx]
        if branch['name'] not in processed:
            if tuple(branch['name'].split('-')) in plot_axis:
                new_branches.append(branch)
                processed.append(branch['name'])
            else:
                next_branch = bandstructure.branches[idx + 1]
                combined = {'start_index': branch['start_index'], 'end_index': next_branch['end_index'], 'name': f'{branch['name'].split('-')[0]}-{next_branch['name'].split('-')[1]}'}
                processed.extend((branch['name'], next_branch['name']))
                new_branches.append(combined)
    for entry in distances_map:
        branch = new_branches[entry[0]]
        if not entry[1]:
            start = branch['start_index']
            stop = branch['end_index'] + 1
            step = 1
        else:
            start = branch['end_index']
            stop = branch['start_index'] - 1 if branch['start_index'] != 0 else None
            step = -1
        new_kpoints += [point.frac_coords for point in bandstructure.kpoints[start:stop:step]]
        for spin in spins:
            for n, band in enumerate(bandstructure.bands[spin]):
                new_bands[spin][n] = np.concatenate((new_bands[spin][n], band[start:stop:step]))
        for spin in spins:
            for n, band in enumerate(bandstructure.projections[spin]):
                new_projections[spin][n] += band[start:stop:step].tolist()
    for spin in spins:
        new_projections[spin] = np.array(new_projections[spin])
    new_labels_dict = {key: point.frac_coords for key, point in bandstructure.labels_dict.items()}
    return BandStructureSymmLine(kpoints=new_kpoints, eigenvals=new_bands, lattice=bandstructure.lattice_rec, efermi=bandstructure.efermi, labels_dict=new_labels_dict, structure=bandstructure.structure, projections=new_projections)