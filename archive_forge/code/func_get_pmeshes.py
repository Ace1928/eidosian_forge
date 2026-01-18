from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def get_pmeshes(self, sites, permutation=None):
    """Returns the pmesh strings used for jmol to show this geometry."""
    pmeshes = []
    _vertices = [site.coords for site in sites] if permutation is None else [sites[ii].coords for ii in permutation]
    _face_centers = []
    n_faces = 0
    for face in self._faces:
        if len(face) in [3, 4]:
            n_faces += 1
        else:
            n_faces += len(face)
        _face_centers.append(np.array([np.mean([_vertices[face_vertex][ii] for face_vertex in face]) for ii in range(3)]))
    out = f'{len(_vertices) + len(_face_centers)}\n'
    for vv in _vertices:
        out += f'{vv[0]:15.8f} {vv[1]:15.8f} {vv[2]:15.8f}\n'
    for fc in _face_centers:
        out += f'{fc[0]:15.8f} {fc[1]:15.8f} {fc[2]:15.8f}\n'
    out += f'{n_faces}\n'
    for iface, face in enumerate(self._faces):
        if len(face) == 3:
            out += '4\n'
        elif len(face) == 4:
            out += '5\n'
        else:
            for ii, f in enumerate(face, start=1):
                out += '4\n'
                out += f'{len(_vertices) + iface}\n'
                out += f'{f}\n'
                out += f'{face[np.mod(ii, len(face))]}\n'
                out += f'{len(_vertices) + iface}\n'
        if len(face) in [3, 4]:
            for face_vertex in face:
                out += f'{face_vertex}\n'
            out += f'{face[0]}\n'
    pmeshes.append({'pmesh_string': out})
    return pmeshes