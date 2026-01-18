from ..featuredetection import TextureFromNoiseImageFilter
def test_TextureFromNoiseImageFilter_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputRadius=dict(argstr='--inputRadius %d'), inputVolume=dict(argstr='--inputVolume %s', extensions=None), outputVolume=dict(argstr='--outputVolume %s', hash_files=False))
    inputs = TextureFromNoiseImageFilter.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value