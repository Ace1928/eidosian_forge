from ..confounds import ComputeDVARS
def test_ComputeDVARS_inputs():
    input_map = dict(figdpi=dict(usedefault=True), figformat=dict(usedefault=True), figsize=dict(usedefault=True), in_file=dict(extensions=None, mandatory=True), in_mask=dict(extensions=None, mandatory=True), intensity_normalization=dict(usedefault=True), remove_zerovariance=dict(usedefault=True), save_all=dict(usedefault=True), save_nstd=dict(usedefault=True), save_plot=dict(usedefault=True), save_std=dict(usedefault=True), save_vxstd=dict(usedefault=True), series_tr=dict(), variance_tol=dict(usedefault=True))
    inputs = ComputeDVARS.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value