from ..anisotropic_power import APMQball
def test_APMQball_inputs():
    input_map = dict(b0_thres=dict(usedefault=True), in_bval=dict(extensions=None, mandatory=True), in_bvec=dict(extensions=None, mandatory=True), in_file=dict(extensions=None, mandatory=True), mask_file=dict(extensions=None), out_prefix=dict())
    inputs = APMQball.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value