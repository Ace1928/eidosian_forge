from ..misc import SimpleThreshold
def test_SimpleThreshold_outputs():
    output_map = dict(thresholded_volumes=dict())
    outputs = SimpleThreshold.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value