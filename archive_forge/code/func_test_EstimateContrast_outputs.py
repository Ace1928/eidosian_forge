from ..model import EstimateContrast
def test_EstimateContrast_outputs():
    output_map = dict(con_images=dict(), ess_images=dict(), spmF_images=dict(), spmT_images=dict(), spm_mat_file=dict(extensions=None))
    outputs = EstimateContrast.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value