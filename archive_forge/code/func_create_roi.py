import os
import os.path as op
import shutil
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ..base import (
from .base import have_cmp
def create_roi(subject_id, subjects_dir, fs_dir, parcellation_name, dilation):
    """Creates the ROI_%s.nii.gz files using the given parcellation information
    from networks. Iteratively create volume."""
    import cmp
    from cmp.util import runCmd
    iflogger.info('Create the ROIs:')
    output_dir = op.abspath(op.curdir)
    fs_dir = op.join(subjects_dir, subject_id)
    cmp_config = cmp.configuration.PipelineConfiguration()
    cmp_config.parcellation_scheme = 'Lausanne2008'
    log = cmp_config.get_logger()
    parval = cmp_config._get_lausanne_parcellation('Lausanne2008')[parcellation_name]
    pgpath = parval['node_information_graphml']
    aseg = nb.load(op.join(fs_dir, 'mri', 'aseg.nii.gz'))
    asegd = np.asanyarray(aseg.dataobj)
    idxr = np.where(asegd == 3)
    idxl = np.where(asegd == 42)
    xx = np.concatenate((idxr[0], idxl[0]))
    yy = np.concatenate((idxr[1], idxl[1]))
    zz = np.concatenate((idxr[2], idxl[2]))
    shape = (25, 25, 25)
    center = np.array(shape) // 2
    dist = np.zeros(shape, dtype='float32')
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                distxyz = center - [x, y, z]
                dist[x, y, z] = np.sqrt(np.sum(np.multiply(distxyz, distxyz)))
    iflogger.info('Working on parcellation: ')
    iflogger.info(cmp_config._get_lausanne_parcellation('Lausanne2008')[parcellation_name])
    iflogger.info('========================')
    pg = nx.read_graphml(pgpath)
    rois = np.zeros((256, 256, 256), dtype=np.int16)
    count = 0
    for brk, brv in pg.nodes(data=True):
        count = count + 1
        iflogger.info(brv)
        iflogger.info(brk)
        if brv['dn_hemisphere'] == 'left':
            hemi = 'lh'
        elif brv['dn_hemisphere'] == 'right':
            hemi = 'rh'
        if brv['dn_region'] == 'subcortical':
            iflogger.info(brv)
            iflogger.info('---------------------')
            iflogger.info('Work on brain region: %s', brv['dn_region'])
            iflogger.info('Freesurfer Name: %s', brv['dn_fsname'])
            iflogger.info('Region %s of %s', count, pg.number_of_nodes())
            iflogger.info('---------------------')
            idx = np.where(asegd == int(brv['dn_fs_aseg_val']))
            rois[idx] = int(brv['dn_correspondence_id'])
        elif brv['dn_region'] == 'cortical':
            iflogger.info(brv)
            iflogger.info('---------------------')
            iflogger.info('Work on brain region: %s', brv['dn_region'])
            iflogger.info('Freesurfer Name: %s', brv['dn_fsname'])
            iflogger.info('Region %s of %s', count, pg.number_of_nodes())
            iflogger.info('---------------------')
            labelpath = op.join(output_dir, parval['fs_label_subdir_name'] % hemi)
            fname = '%s.%s.label' % (hemi, brv['dn_fsname'])
            mri_cmd = 'mri_label2vol --label "%s" --temp "%s" --o "%s" --identity' % (op.join(labelpath, fname), op.join(fs_dir, 'mri', 'orig.mgz'), op.join(output_dir, 'tmp.nii.gz'))
            runCmd(mri_cmd, log)
            tmp = nb.load(op.join(output_dir, 'tmp.nii.gz'))
            tmpd = np.asanyarray(tmp.dataobj)
            idx = np.where(tmpd == 1)
            rois[idx] = int(brv['dn_correspondence_id'])
        out_roi = op.abspath('ROI_%s.nii.gz' % parcellation_name)
        hdr = aseg.header
        hdr2 = hdr.copy()
        hdr2.set_data_dtype(np.uint16)
        log.info('Save output image to %s' % out_roi)
        img = nb.Nifti1Image(rois, aseg.affine, hdr2)
        nb.save(img, out_roi)
    iflogger.info('[ DONE ]')
    if dilation is True:
        iflogger.info('Dilating cortical regions...')
        for j in range(xx.size):
            if rois[xx[j], yy[j], zz[j]] == 0:
                local = extract(rois, shape, position=(xx[j], yy[j], zz[j]), fill=0)
                mask = local.copy()
                mask[np.nonzero(local > 0)] = 1
                thisdist = np.multiply(dist, mask)
                thisdist[np.nonzero(thisdist == 0)] = np.amax(thisdist)
                value = np.int_(local[np.nonzero(thisdist == np.amin(thisdist))])
                if value.size > 1:
                    counts = np.bincount(value)
                    value = np.argmax(counts)
                rois[xx[j], yy[j], zz[j]] = value
        out_roi = op.abspath('ROIv_%s.nii.gz' % parcellation_name)
        iflogger.info('Save output image to %s', out_roi)
        img = nb.Nifti1Image(rois, aseg.affine, hdr2)
        nb.save(img, out_roi)
        iflogger.info('[ DONE ]')