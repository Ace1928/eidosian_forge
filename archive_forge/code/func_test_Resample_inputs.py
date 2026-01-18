from ..preprocess import Resample
def test_Resample_inputs():
    input_map = dict(in_file=dict(extensions=None, mandatory=True), interp=dict(mandatory=True, usedefault=True), vox_size=dict())
    inputs = Resample.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value