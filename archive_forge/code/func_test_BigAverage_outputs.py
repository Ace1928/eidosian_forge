from ..minc import BigAverage
def test_BigAverage_outputs():
    output_map = dict(output_file=dict(extensions=None), sd_file=dict(extensions=None))
    outputs = BigAverage.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value