import os
import os.path as op
import shutil
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ..base import (
from .base import have_cmp
def create_annot_label(subject_id, subjects_dir, fs_dir, parcellation_name):
    import cmp
    from cmp.util import runCmd
    iflogger.info('Create the cortical labels necessary for our ROIs')
    iflogger.info('=================================================')
    fs_label_dir = op.join(op.join(subjects_dir, subject_id), 'label')
    output_dir = op.abspath(op.curdir)
    paths = []
    cmp_config = cmp.configuration.PipelineConfiguration()
    cmp_config.parcellation_scheme = 'Lausanne2008'
    for hemi in ['lh', 'rh']:
        spath = cmp_config._get_lausanne_parcellation('Lausanne2008')[parcellation_name]['fs_label_subdir_name'] % hemi
        paths.append(spath)
    for p in paths:
        try:
            os.makedirs(op.join('.', p))
        except:
            pass
    if '33' in parcellation_name:
        comp = [('rh', 'myatlas_36_rh.gcs', 'rh.myaparc_36.annot', 'regenerated_rh_36', 'myaparc_36'), ('rh', 'myatlas_60_rh.gcs', 'rh.myaparc_60.annot', 'regenerated_rh_60', 'myaparc_60'), ('lh', 'myatlas_36_lh.gcs', 'lh.myaparc_36.annot', 'regenerated_lh_36', 'myaparc_36'), ('lh', 'myatlas_60_lh.gcs', 'lh.myaparc_60.annot', 'regenerated_lh_60', 'myaparc_60')]
    elif '60' in parcellation_name:
        comp = [('rh', 'myatlas_60_rh.gcs', 'rh.myaparc_60.annot', 'regenerated_rh_60', 'myaparc_60'), ('lh', 'myatlas_60_lh.gcs', 'lh.myaparc_60.annot', 'regenerated_lh_60', 'myaparc_60')]
    elif '125' in parcellation_name:
        comp = [('rh', 'myatlas_125_rh.gcs', 'rh.myaparc_125.annot', 'regenerated_rh_125', 'myaparc_125'), ('rh', 'myatlas_60_rh.gcs', 'rh.myaparc_60.annot', 'regenerated_rh_60', 'myaparc_60'), ('lh', 'myatlas_125_lh.gcs', 'lh.myaparc_125.annot', 'regenerated_lh_125', 'myaparc_125'), ('lh', 'myatlas_60_lh.gcs', 'lh.myaparc_60.annot', 'regenerated_lh_60', 'myaparc_60')]
    elif '250' in parcellation_name:
        comp = [('rh', 'myatlas_250_rh.gcs', 'rh.myaparc_250.annot', 'regenerated_rh_250', 'myaparc_250'), ('rh', 'myatlas_60_rh.gcs', 'rh.myaparc_60.annot', 'regenerated_rh_60', 'myaparc_60'), ('lh', 'myatlas_250_lh.gcs', 'lh.myaparc_250.annot', 'regenerated_lh_250', 'myaparc_250'), ('lh', 'myatlas_60_lh.gcs', 'lh.myaparc_60.annot', 'regenerated_lh_60', 'myaparc_60')]
    else:
        comp = [('rh', 'myatlas_36_rh.gcs', 'rh.myaparc_36.annot', 'regenerated_rh_36', 'myaparc_36'), ('rh', 'myatlasP1_16_rh.gcs', 'rh.myaparcP1_16.annot', 'regenerated_rh_500', 'myaparcP1_16'), ('rh', 'myatlasP17_28_rh.gcs', 'rh.myaparcP17_28.annot', 'regenerated_rh_500', 'myaparcP17_28'), ('rh', 'myatlasP29_36_rh.gcs', 'rh.myaparcP29_36.annot', 'regenerated_rh_500', 'myaparcP29_36'), ('rh', 'myatlas_60_rh.gcs', 'rh.myaparc_60.annot', 'regenerated_rh_60', 'myaparc_60'), ('rh', 'myatlas_125_rh.gcs', 'rh.myaparc_125.annot', 'regenerated_rh_125', 'myaparc_125'), ('rh', 'myatlas_250_rh.gcs', 'rh.myaparc_250.annot', 'regenerated_rh_250', 'myaparc_250'), ('lh', 'myatlas_36_lh.gcs', 'lh.myaparc_36.annot', 'regenerated_lh_36', 'myaparc_36'), ('lh', 'myatlasP1_16_lh.gcs', 'lh.myaparcP1_16.annot', 'regenerated_lh_500', 'myaparcP1_16'), ('lh', 'myatlasP17_28_lh.gcs', 'lh.myaparcP17_28.annot', 'regenerated_lh_500', 'myaparcP17_28'), ('lh', 'myatlasP29_36_lh.gcs', 'lh.myaparcP29_36.annot', 'regenerated_lh_500', 'myaparcP29_36'), ('lh', 'myatlas_60_lh.gcs', 'lh.myaparc_60.annot', 'regenerated_lh_60', 'myaparc_60'), ('lh', 'myatlas_125_lh.gcs', 'lh.myaparc_125.annot', 'regenerated_lh_125', 'myaparc_125'), ('lh', 'myatlas_250_lh.gcs', 'lh.myaparc_250.annot', 'regenerated_lh_250', 'myaparc_250')]
    log = cmp_config.get_logger()
    for out in comp:
        mris_cmd = 'mris_ca_label %s %s "%s/surf/%s.sphere.reg" "%s" "%s" ' % (subject_id, out[0], op.join(subjects_dir, subject_id), out[0], cmp_config.get_lausanne_atlas(out[1]), op.join(fs_label_dir, out[2]))
        runCmd(mris_cmd, log)
        iflogger.info('-----------')
        annot = '--annotation "%s"' % out[4]
        mri_an_cmd = 'mri_annotation2label --subject %s --hemi %s --outdir "%s" %s' % (subject_id, out[0], op.join(output_dir, out[3]), annot)
        iflogger.info(mri_an_cmd)
        runCmd(mri_an_cmd, log)
        iflogger.info('-----------')
        iflogger.info(os.environ['SUBJECTS_DIR'])
        rhun = op.join(output_dir, 'rh.unknown.label')
        lhun = op.join(output_dir, 'lh.unknown.label')
        rhco = op.join(output_dir, 'rh.corpuscallosum.label')
        lhco = op.join(output_dir, 'lh.corpuscallosum.label')
    shutil.copy(op.join(output_dir, 'regenerated_rh_60', 'rh.unknown.label'), rhun)
    shutil.copy(op.join(output_dir, 'regenerated_lh_60', 'lh.unknown.label'), lhun)
    shutil.copy(op.join(output_dir, 'regenerated_rh_60', 'rh.corpuscallosum.label'), rhco)
    shutil.copy(op.join(output_dir, 'regenerated_lh_60', 'lh.corpuscallosum.label'), lhco)
    mri_cmd = 'mri_label2vol --label "%s" --label "%s" --label "%s" --label "%s" --temp "%s" --o  "%s" --identity ' % (rhun, lhun, rhco, lhco, op.join(op.join(subjects_dir, subject_id), 'mri', 'orig.mgz'), op.join(fs_label_dir, 'cc_unknown.nii.gz'))
    runCmd(mri_cmd, log)
    runCmd('mris_volmask %s' % subject_id, log)
    mri_cmd = 'mri_convert -i "%s/mri/ribbon.mgz" -o "%s/mri/ribbon.nii.gz"' % (op.join(subjects_dir, subject_id), op.join(subjects_dir, subject_id))
    runCmd(mri_cmd, log)
    mri_cmd = 'mri_convert -i "%s/mri/aseg.mgz" -o "%s/mri/aseg.nii.gz"' % (op.join(subjects_dir, subject_id), op.join(subjects_dir, subject_id))
    runCmd(mri_cmd, log)
    iflogger.info('[ DONE ]')