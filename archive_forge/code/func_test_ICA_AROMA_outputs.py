from ..aroma import ICA_AROMA
def test_ICA_AROMA_outputs():
    output_map = dict(aggr_denoised_file=dict(extensions=None), nonaggr_denoised_file=dict(extensions=None), out_dir=dict())
    outputs = ICA_AROMA.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value