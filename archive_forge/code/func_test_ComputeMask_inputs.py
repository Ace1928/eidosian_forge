from ..preprocess import ComputeMask
def test_ComputeMask_inputs():
    input_map = dict(M=dict(), cc=dict(), m=dict(), mean_volume=dict(extensions=None, mandatory=True), reference_volume=dict(extensions=None))
    inputs = ComputeMask.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value