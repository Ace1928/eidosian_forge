from ..io import JSONFileSink
def test_JSONFileSink_inputs():
    input_map = dict(_outputs=dict(usedefault=True), in_dict=dict(usedefault=True), out_file=dict(extensions=None))
    inputs = JSONFileSink.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value