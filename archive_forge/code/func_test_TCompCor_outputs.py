from ..confounds import TCompCor
def test_TCompCor_outputs():
    output_map = dict(components_file=dict(extensions=None), high_variance_masks=dict(), metadata_file=dict(extensions=None), pre_filter_file=dict(extensions=None))
    outputs = TCompCor.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value