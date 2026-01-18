from ..brains import CleanUpOverlapLabels
def test_CleanUpOverlapLabels_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputBinaryVolumes=dict(argstr='--inputBinaryVolumes %s...'), outputBinaryVolumes=dict(argstr='--outputBinaryVolumes %s...', hash_files=False))
    inputs = CleanUpOverlapLabels.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value