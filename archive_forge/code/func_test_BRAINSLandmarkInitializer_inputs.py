from ..brains import BRAINSLandmarkInitializer
def test_BRAINSLandmarkInitializer_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputFixedLandmarkFilename=dict(argstr='--inputFixedLandmarkFilename %s', extensions=None), inputMovingLandmarkFilename=dict(argstr='--inputMovingLandmarkFilename %s', extensions=None), inputWeightFilename=dict(argstr='--inputWeightFilename %s', extensions=None), outputTransformFilename=dict(argstr='--outputTransformFilename %s', hash_files=False))
    inputs = BRAINSLandmarkInitializer.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value