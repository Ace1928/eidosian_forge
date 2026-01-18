import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def read_annot(filepath, orig_ids=False):
    """Read in a Freesurfer annotation from a ``.annot`` file.

    An ``.annot`` file contains a sequence of vertices with a label (also known
    as an "annotation value") associated with each vertex, and then a sequence
    of colors corresponding to each label.

    Annotation file format versions 1 and 2 are supported, corresponding to
    the "old-style" and "new-style" color table layout.

    Note that the output color table ``ctab`` is in RGBT form, where T
    (transparency) is 255 - alpha.

    See:
     * https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles#Annotation
     * https://github.com/freesurfer/freesurfer/blob/dev/matlab/read_annotation.m
     * https://github.com/freesurfer/freesurfer/blob/8b88b34/utils/colortab.c

    Parameters
    ----------
    filepath : str
        Path to annotation file.
    orig_ids : bool
        Whether to return the vertex ids as stored in the annotation
        file or the positional colortable ids. With orig_ids=False
        vertices with no id have an id set to -1.

    Returns
    -------
    labels : ndarray, shape (n_vertices,)
        Annotation id at each vertex. If a vertex does not belong
        to any label and orig_ids=False, its id will be set to -1.
    ctab : ndarray, shape (n_labels, 5)
        RGBT + label id colortable array.
    names : list of bytes
        The names of the labels. The length of the list is n_labels.
    """
    with open(filepath, 'rb') as fobj:
        dt = _ANNOT_DT
        vnum = np.fromfile(fobj, dt, 1)[0]
        data = np.fromfile(fobj, dt, vnum * 2).reshape(vnum, 2)
        labels = data[:, 1]
        ctab_exists = np.fromfile(fobj, dt, 1)[0]
        if not ctab_exists:
            raise Exception('Color table not found in annotation file')
        n_entries = np.fromfile(fobj, dt, 1)[0]
        if n_entries > 0:
            ctab, names = _read_annot_ctab_old_format(fobj, n_entries)
        else:
            ctab, names = _read_annot_ctab_new_format(fobj, -n_entries)
    ctab[:, [4]] = _pack_rgb(ctab[:, :3])
    if not orig_ids:
        ord = np.argsort(ctab[:, -1])
        mask = labels != 0
        labels[~mask] = -1
        labels[mask] = ord[np.searchsorted(ctab[ord, -1], labels[mask])]
    return (labels, ctab, names)