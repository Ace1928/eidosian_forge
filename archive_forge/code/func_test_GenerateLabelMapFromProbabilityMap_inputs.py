from ..brains import GenerateLabelMapFromProbabilityMap
def test_GenerateLabelMapFromProbabilityMap_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputVolumes=dict(argstr='--inputVolumes %s...'), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputLabelVolume=dict(argstr='--outputLabelVolume %s', hash_files=False))
    inputs = GenerateLabelMapFromProbabilityMap.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value