import os
import os.path as op
import shutil
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ..base import (
from .base import have_cmp
def create_wm_mask(subject_id, subjects_dir, fs_dir, parcellation_name):
    import cmp
    import scipy.ndimage.morphology as nd
    iflogger.info('Create white matter mask')
    fs_dir = op.join(subjects_dir, subject_id)
    cmp_config = cmp.configuration.PipelineConfiguration()
    cmp_config.parcellation_scheme = 'Lausanne2008'
    pgpath = cmp_config._get_lausanne_parcellation('Lausanne2008')[parcellation_name]['node_information_graphml']
    fsmask = nb.load(op.join(fs_dir, 'mri', 'ribbon.nii.gz'))
    fsmaskd = np.asanyarray(fsmask.dataobj)
    wmmask = np.zeros(fsmaskd.shape)
    idx_lh = np.where(fsmaskd == 120)
    idx_rh = np.where(fsmaskd == 20)
    wmmask[idx_lh] = 1
    wmmask[idx_rh] = 1
    aseg = nb.load(op.join(fs_dir, 'mri', 'aseg.nii.gz'))
    asegd = np.asanyarray(aseg.dataobj)
    imerode = nd.binary_erosion
    csfA = np.zeros(asegd.shape)
    csfB = np.zeros(asegd.shape)
    se1 = np.zeros((3, 3, 5))
    se1[1, :, 2] = 1
    se1[:, 1, 2] = 1
    se1[1, 1, :] = 1
    se = np.zeros((3, 3, 3))
    se[1, :, 1] = 1
    se[:, 1, 1] = 1
    se[1, 1, :] = 1
    idx = np.where((asegd == 4) | (asegd == 43) | (asegd == 11) | (asegd == 50) | (asegd == 31) | (asegd == 63) | (asegd == 10) | (asegd == 49))
    csfA[idx] = 1
    csfA = imerode(imerode(csfA, se1), se)
    idx = np.where((asegd == 11) | (asegd == 50) | (asegd == 10) | (asegd == 49))
    csfA[idx] = 0
    idx = np.where((asegd == 5) | (asegd == 14) | (asegd == 15) | (asegd == 24) | (asegd == 44) | (asegd == 72) | (asegd == 75) | (asegd == 76) | (asegd == 213) | (asegd == 221))
    for i in [5, 14, 15, 24, 44, 72, 75, 76, 213, 221]:
        idx = np.where(asegd == i)
        csfB[idx] = 1
    gr_ncl = np.zeros(asegd.shape)
    for i in [10, 11, 12, 49, 50, 51]:
        idx = np.where(asegd == i)
        tmp = np.zeros(asegd.shape)
        tmp[idx] = 1
        tmp = imerode(tmp, se)
        idx = np.where(tmp == 1)
        gr_ncl[idx] = 1
    for i in [13, 17, 18, 26, 52, 53, 54, 58]:
        idx = np.where(asegd == i)
        gr_ncl[idx] = 1
    remaining = np.zeros(asegd.shape)
    idx = np.where(asegd == 16)
    remaining[idx] = 1
    idx = np.where((csfA != 0) | (csfB != 0) | (gr_ncl != 0) | (remaining != 0))
    wmmask[idx] = 0
    iflogger.info('Removing lateral ventricles and eroded grey nuclei and brainstem from white matter mask')
    ccun = nb.load(op.join(fs_dir, 'label', 'cc_unknown.nii.gz'))
    ccund = np.asanyarray(ccun.dataobj)
    idx = np.where(ccund != 0)
    iflogger.info('Add corpus callosum and unknown to wm mask')
    wmmask[idx] = 1
    iflogger.info('Loading ROI_%s.nii.gz to subtract cortical ROIs from white matter mask', parcellation_name)
    roi = nb.load(op.join(op.curdir, 'ROI_%s.nii.gz' % parcellation_name))
    roid = np.asanyarray(roi.dataobj)
    assert roid.shape[0] == wmmask.shape[0]
    pg = nx.read_graphml(pgpath)
    for brk, brv in pg.nodes(data=True):
        if brv['dn_region'] == 'cortical':
            iflogger.info('Subtracting region %s with intensity value %s', brv['dn_region'], brv['dn_correspondence_id'])
            idx = np.where(roid == int(brv['dn_correspondence_id']))
            wmmask[idx] = 0
    wm_out = op.join(fs_dir, 'mri', 'fsmask_1mm.nii.gz')
    img = nb.Nifti1Image(wmmask, fsmask.affine, fsmask.header)
    iflogger.info('Save white matter mask: %s', wm_out)
    nb.save(img, wm_out)