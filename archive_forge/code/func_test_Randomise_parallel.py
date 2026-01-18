import os
import nipype.interfaces.fsl.dti as fsl
from nipype.interfaces.fsl import Info, no_fsl
from nipype.interfaces.base import Undefined
import pytest
from nipype.testing.fixtures import create_files_in_directory
@pytest.mark.xfail(reason='These tests are skipped until we clean up some of this code')
def test_Randomise_parallel():
    rand = fsl.Randomise_parallel()
    assert rand.cmd == 'randomise_parallel'
    with pytest.raises(ValueError):
        rand.run()
    rand.inputs.input_4D = 'infile.nii'
    rand.inputs.output_rootname = 'outfile'
    rand.inputs.design_matrix = 'design.mat'
    rand.inputs.t_contrast = 'infile.con'
    actualCmdline = sorted(rand.cmdline.split())
    cmd = 'randomise_parallel -i infile.nii -o outfile -d design.mat -t infile.con'
    desiredCmdline = sorted(cmd.split())
    assert actualCmdline == desiredCmdline
    rand2 = fsl.Randomise_parallel(input_4D='infile2', output_rootname='outfile2', f_contrast='infile.f', one_sample_gmean=True, int_seed=4)
    actualCmdline = sorted(rand2.cmdline.split())
    cmd = 'randomise_parallel -i infile2 -o outfile2 -1 -f infile.f --seed=4'
    desiredCmdline = sorted(cmd.split())
    assert actualCmdline == desiredCmdline
    rand3 = fsl.Randomise_parallel()
    results = rand3.run(input_4D='infile3', output_rootname='outfile3')
    assert results.runtime.cmdline == 'randomise_parallel -i infile3 -o outfile3'
    opt_map = {'demean_data': ('-D', True), 'one_sample_gmean': ('-1', True), 'mask_image': ('-m inp_mask', 'inp_mask'), 'design_matrix': ('-d design.mat', 'design.mat'), 't_contrast': ('-t input.con', 'input.con'), 'f_contrast': ('-f input.fts', 'input.fts'), 'xchange_block_labels': ('-e design.grp', 'design.grp'), 'print_unique_perm': ('-q', True), 'print_info_parallelMode': ('-Q', True), 'num_permutations': ('-n 10', 10), 'vox_pvalus': ('-x', True), 'fstats_only': ('--fonly', True), 'thresh_free_cluster': ('-T', True), 'thresh_free_cluster_2Dopt': ('--T2', True), 'cluster_thresholding': ('-c 0.20', 0.2), 'cluster_mass_thresholding': ('-C 0.40', 0.4), 'fcluster_thresholding': ('-F 0.10', 0.1), 'fcluster_mass_thresholding': ('-S 0.30', 0.3), 'variance_smoothing': ('-v 0.20', 0.2), 'diagnostics_off': ('--quiet', True), 'output_raw': ('-R', True), 'output_perm_vect': ('-P', True), 'int_seed': ('--seed=20', 20), 'TFCE_height_param': ('--tfce_H=0.11', 0.11), 'TFCE_extent_param': ('--tfce_E=0.50', 0.5), 'TFCE_connectivity': ('--tfce_C=0.30', 0.3), 'list_num_voxel_EVs_pos': ('--vxl=' + repr([1, 2, 3, 4]), repr([1, 2, 3, 4])), 'list_img_voxel_EVs': ('--vxf=' + repr([6, 7, 8, 9, 3]), repr([6, 7, 8, 9, 3]))}
    for name, settings in list(opt_map.items()):
        rand4 = fsl.Randomise_parallel(input_4D='infile', output_rootname='root', **{name: settings[1]})
        assert rand4.cmdline == rand4.cmd + ' -i infile -o root ' + settings[0]