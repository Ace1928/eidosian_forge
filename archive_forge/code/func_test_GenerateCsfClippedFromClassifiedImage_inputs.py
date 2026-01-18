from ..featurecreator import GenerateCsfClippedFromClassifiedImage
def test_GenerateCsfClippedFromClassifiedImage_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputCassifiedVolume=dict(argstr='--inputCassifiedVolume %s', extensions=None), outputVolume=dict(argstr='--outputVolume %s', hash_files=False))
    inputs = GenerateCsfClippedFromClassifiedImage.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value