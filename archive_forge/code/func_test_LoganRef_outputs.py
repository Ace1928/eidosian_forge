from ..petsurfer import LoganRef
def test_LoganRef_outputs():
    output_map = dict(beta_file=dict(extensions=None), bp_file=dict(extensions=None), dof_file=dict(extensions=None), error_file=dict(extensions=None), error_stddev_file=dict(extensions=None), error_var_file=dict(extensions=None), estimate_file=dict(extensions=None), frame_eigenvectors=dict(extensions=None), ftest_file=dict(), fwhm_file=dict(extensions=None), gamma_file=dict(), gamma_var_file=dict(), glm_dir=dict(), k2p_file=dict(extensions=None), mask_file=dict(extensions=None), sig_file=dict(), singular_values=dict(extensions=None), spatial_eigenvectors=dict(extensions=None), svd_stats_file=dict(extensions=None))
    outputs = LoganRef.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value