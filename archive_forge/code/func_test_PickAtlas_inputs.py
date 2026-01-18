from ..misc import PickAtlas
def test_PickAtlas_inputs():
    input_map = dict(atlas=dict(extensions=None, mandatory=True), dilation_size=dict(usedefault=True), hemi=dict(usedefault=True), labels=dict(mandatory=True), output_file=dict(extensions=None))
    inputs = PickAtlas.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value