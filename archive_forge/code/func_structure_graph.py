from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
def structure_graph(self, include_critical_points=('bond', 'ring', 'cage')):
    """A StructureGraph object describing bonding information in the crystal.

        Args:
            include_critical_points: add DummySpecies for the critical points themselves, a list of
                "nucleus", "bond", "ring", "cage", set to None to disable

        Returns:
            StructureGraph
        """
    structure = self.structure.copy()
    point_idx_to_struct_idx = {}
    if include_critical_points:
        for prop in ('ellipticity', 'laplacian', 'field'):
            structure.add_site_property(prop, [0] * len(structure))
        for idx, node in self.nodes.items():
            cp = self.critical_points[node['unique_idx']]
            if cp.type.value in include_critical_points:
                specie = DummySpecies(f'X{cp.type.value[0]}cp', oxidation_state=None)
                structure.append(specie, node['frac_coords'], properties={'ellipticity': cp.ellipticity, 'laplacian': cp.laplacian, 'field': cp.field})
                point_idx_to_struct_idx[idx] = len(structure) - 1
    edge_weight = 'bond_length'
    edge_weight_units = 'Å'
    struct_graph = StructureGraph.from_empty_graph(structure, name='bonds', edge_weight_name=edge_weight, edge_weight_units=edge_weight_units)
    edges = self.edges.copy()
    idx_to_delete = []
    for idx, edge in edges.items():
        unique_idx = self.nodes[idx]['unique_idx']
        if self.critical_points[unique_idx].type == CriticalPointType.bond and idx not in idx_to_delete:
            for idx2, edge2 in edges.items():
                if idx != idx2 and edge == edge2:
                    idx_to_delete.append(idx2)
                    warnings.warn('Duplicate edge detected, try re-running critic2 with custom parameters to fix this. Mostly harmless unless user is also interested in rings/cages.')
                    logger.debug(f'Duplicate edge between points {idx} (unique point {self.nodes[idx]['unique_idx']})and {idx2} ({self.nodes[idx2]['unique_idx']}).')
    for idx in idx_to_delete:
        del edges[idx]
    for idx, edge in edges.items():
        unique_idx = self.nodes[idx]['unique_idx']
        if self.critical_points[unique_idx].type == CriticalPointType.bond:
            from_idx = edge['from_idx']
            to_idx = edge['to_idx']
            skip_bond = False
            if include_critical_points and 'nnattr' not in include_critical_points:
                from_type = self.critical_points[self.nodes[from_idx]['unique_idx']].type
                to_type = self.critical_points[self.nodes[from_idx]['unique_idx']].type
                skip_bond = from_type != CriticalPointType.nucleus or to_type != CriticalPointType.nucleus
            if not skip_bond:
                from_lvec = edge['from_lvec']
                to_lvec = edge['to_lvec']
                relative_lvec = np.subtract(to_lvec, from_lvec)
                struct_from_idx = point_idx_to_struct_idx.get(from_idx, from_idx)
                struct_to_idx = point_idx_to_struct_idx.get(to_idx, to_idx)
                weight = self.structure.get_distance(struct_from_idx, struct_to_idx, jimage=relative_lvec)
                crit_point = self.critical_points[unique_idx]
                edge_properties = {'field': crit_point.field, 'laplacian': crit_point.laplacian, 'ellipticity': crit_point.ellipticity, 'frac_coords': self.nodes[idx]['frac_coords']}
                struct_graph.add_edge(struct_from_idx, struct_to_idx, from_jimage=from_lvec, to_jimage=to_lvec, weight=weight, edge_properties=edge_properties)
    return struct_graph