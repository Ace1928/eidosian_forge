from ..metric import MetricResample
def test_MetricResample_outputs():
    output_map = dict(out_file=dict(extensions=None), roi_file=dict(extensions=None))
    outputs = MetricResample.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value