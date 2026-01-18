from ..convert import MergeCNetworks
def test_MergeCNetworks_inputs():
    input_map = dict(in_files=dict(mandatory=True), out_file=dict(extensions=None, usedefault=True))
    inputs = MergeCNetworks.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value