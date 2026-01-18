from ..preprocess import OutlierCount
def test_OutlierCount_outputs():
    output_map = dict(out_file=dict(extensions=None), out_outliers=dict(extensions=None))
    outputs = OutlierCount.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value