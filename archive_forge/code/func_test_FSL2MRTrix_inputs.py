from ..tensors import FSL2MRTrix
def test_FSL2MRTrix_inputs():
    input_map = dict(bval_file=dict(extensions=None, mandatory=True), bvec_file=dict(extensions=None, mandatory=True), invert_x=dict(usedefault=True), invert_y=dict(usedefault=True), invert_z=dict(usedefault=True), out_encoding_file=dict(extensions=None, genfile=True))
    inputs = FSL2MRTrix.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value