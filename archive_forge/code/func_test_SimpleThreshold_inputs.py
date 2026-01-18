from ..misc import SimpleThreshold
def test_SimpleThreshold_inputs():
    input_map = dict(threshold=dict(mandatory=True), volumes=dict(mandatory=True))
    inputs = SimpleThreshold.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value