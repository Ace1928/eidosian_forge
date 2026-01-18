from ..regutils import RegAverage
def test_RegAverage_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = RegAverage.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value