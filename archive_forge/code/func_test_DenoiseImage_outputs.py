from ..segmentation import DenoiseImage
def test_DenoiseImage_outputs():
    output_map = dict(noise_image=dict(extensions=None), output_image=dict(extensions=None))
    outputs = DenoiseImage.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value