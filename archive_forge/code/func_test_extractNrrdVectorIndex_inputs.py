from ..gtract import extractNrrdVectorIndex
def test_extractNrrdVectorIndex_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputVolume=dict(argstr='--inputVolume %s', extensions=None), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputVolume=dict(argstr='--outputVolume %s', hash_files=False), setImageOrientation=dict(argstr='--setImageOrientation %s'), vectorIndex=dict(argstr='--vectorIndex %d'))
    inputs = extractNrrdVectorIndex.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value