from ..gtract import gtractConcatDwi
def test_gtractConcatDwi_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), ignoreOrigins=dict(argstr='--ignoreOrigins '), inputVolume=dict(argstr='--inputVolume %s...'), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputVolume=dict(argstr='--outputVolume %s', hash_files=False))
    inputs = gtractConcatDwi.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value