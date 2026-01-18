from ..gtract import gtractAverageBvalues
def test_gtractAverageBvalues_inputs():
    input_map = dict(args=dict(argstr='%s'), averageB0only=dict(argstr='--averageB0only '), directionsTolerance=dict(argstr='--directionsTolerance %f'), environ=dict(nohash=True, usedefault=True), inputVolume=dict(argstr='--inputVolume %s', extensions=None), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputVolume=dict(argstr='--outputVolume %s', hash_files=False))
    inputs = gtractAverageBvalues.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value