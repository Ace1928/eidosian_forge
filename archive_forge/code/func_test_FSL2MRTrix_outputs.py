from ..tensors import FSL2MRTrix
def test_FSL2MRTrix_outputs():
    output_map = dict(encoding_file=dict(extensions=None))
    outputs = FSL2MRTrix.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value