from ..brains import BRAINSTransformConvert
def test_BRAINSTransformConvert_inputs():
    input_map = dict(args=dict(argstr='%s'), displacementVolume=dict(argstr='--displacementVolume %s', hash_files=False), environ=dict(nohash=True, usedefault=True), inputTransform=dict(argstr='--inputTransform %s', extensions=None), outputPrecisionType=dict(argstr='--outputPrecisionType %s'), outputTransform=dict(argstr='--outputTransform %s', hash_files=False), outputTransformType=dict(argstr='--outputTransformType %s'), referenceVolume=dict(argstr='--referenceVolume %s', extensions=None))
    inputs = BRAINSTransformConvert.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value