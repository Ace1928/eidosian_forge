import os
import os.path as op
import shutil
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ..base import (
from .base import have_cmp
def crop_and_move_datasets(subject_id, subjects_dir, fs_dir, parcellation_name, out_roi_file, dilation):
    from cmp.util import runCmd
    fs_dir = op.join(subjects_dir, subject_id)
    cmp_config = cmp.configuration.PipelineConfiguration()
    cmp_config.parcellation_scheme = 'Lausanne2008'
    log = cmp_config.get_logger()
    output_dir = op.abspath(op.curdir)
    iflogger.info('Cropping and moving datasets to %s', output_dir)
    ds = [(op.join(fs_dir, 'mri', 'aseg.nii.gz'), op.abspath('aseg.nii.gz')), (op.join(fs_dir, 'mri', 'ribbon.nii.gz'), op.abspath('ribbon.nii.gz')), (op.join(fs_dir, 'mri', 'fsmask_1mm.nii.gz'), op.abspath('fsmask_1mm.nii.gz')), (op.join(fs_dir, 'label', 'cc_unknown.nii.gz'), op.abspath('cc_unknown.nii.gz'))]
    ds.append((op.abspath('ROI_%s.nii.gz' % parcellation_name), op.abspath('ROI_HR_th.nii.gz')))
    if dilation is True:
        ds.append((op.abspath('ROIv_%s.nii.gz' % parcellation_name), op.abspath('ROIv_HR_th.nii.gz')))
    orig = op.join(fs_dir, 'mri', 'orig', '001.mgz')
    for d in ds:
        iflogger.info('Processing %s:', d[0])
        if not op.exists(d[0]):
            raise Exception('File %s does not exist.' % d[0])
        mri_cmd = 'mri_convert -rl "%s" -rt nearest "%s" -nc "%s"' % (orig, d[0], d[1])
        runCmd(mri_cmd, log)