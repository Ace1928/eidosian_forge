from ..preprocess import NetCorr
def test_NetCorr_outputs():
    output_map = dict(out_corr_maps=dict(), out_corr_matrix=dict(extensions=None))
    outputs = NetCorr.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value