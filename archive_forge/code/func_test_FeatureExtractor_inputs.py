from ..fix import FeatureExtractor
def test_FeatureExtractor_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), mel_ica=dict(argstr='%s', copyfile=False, position=-1))
    inputs = FeatureExtractor.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value