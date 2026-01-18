from ..preprocess import Denoise
def test_Denoise_inputs():
    input_map = dict(block_radius=dict(usedefault=True), in_file=dict(extensions=None, mandatory=True), in_mask=dict(extensions=None), noise_mask=dict(extensions=None), noise_model=dict(mandatory=True, usedefault=True), patch_radius=dict(usedefault=True), signal_mask=dict(extensions=None), snr=dict())
    inputs = Denoise.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value