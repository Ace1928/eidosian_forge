from ..brainsresample import BRAINSResample
def test_BRAINSResample_inputs():
    input_map = dict(args=dict(argstr='%s'), defaultValue=dict(argstr='--defaultValue %f'), deformationVolume=dict(argstr='--deformationVolume %s', extensions=None), environ=dict(nohash=True, usedefault=True), gridSpacing=dict(argstr='--gridSpacing %s', sep=','), inputVolume=dict(argstr='--inputVolume %s', extensions=None), interpolationMode=dict(argstr='--interpolationMode %s'), inverseTransform=dict(argstr='--inverseTransform '), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputVolume=dict(argstr='--outputVolume %s', hash_files=False), pixelType=dict(argstr='--pixelType %s'), referenceVolume=dict(argstr='--referenceVolume %s', extensions=None), warpTransform=dict(argstr='--warpTransform %s', extensions=None))
    inputs = BRAINSResample.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value