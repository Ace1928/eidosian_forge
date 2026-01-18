from ..brains import ShuffleVectorsModule
def test_ShuffleVectorsModule_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputVectorFileBaseName=dict(argstr='--inputVectorFileBaseName %s', extensions=None), outputVectorFileBaseName=dict(argstr='--outputVectorFileBaseName %s', hash_files=False), resampleProportion=dict(argstr='--resampleProportion %f'))
    inputs = ShuffleVectorsModule.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value