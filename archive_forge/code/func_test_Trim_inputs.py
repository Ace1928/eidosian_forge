from ..preprocess import Trim
def test_Trim_inputs():
    input_map = dict(begin_index=dict(usedefault=True), end_index=dict(usedefault=True), in_file=dict(extensions=None, mandatory=True), out_file=dict(extensions=None), suffix=dict(usedefault=True))
    inputs = Trim.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value