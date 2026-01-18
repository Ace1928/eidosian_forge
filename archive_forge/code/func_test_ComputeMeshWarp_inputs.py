from ..mesh import ComputeMeshWarp
def test_ComputeMeshWarp_inputs():
    input_map = dict(metric=dict(usedefault=True), out_file=dict(extensions=None, usedefault=True), out_warp=dict(extensions=None, usedefault=True), surface1=dict(extensions=None, mandatory=True), surface2=dict(extensions=None, mandatory=True), weighting=dict(usedefault=True))
    inputs = ComputeMeshWarp.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value