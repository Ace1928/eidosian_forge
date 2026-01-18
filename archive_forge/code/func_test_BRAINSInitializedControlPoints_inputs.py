from ..brains import BRAINSInitializedControlPoints
def test_BRAINSInitializedControlPoints_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputVolume=dict(argstr='--inputVolume %s', extensions=None), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputLandmarksFile=dict(argstr='--outputLandmarksFile %s'), outputVolume=dict(argstr='--outputVolume %s', hash_files=False), permuteOrder=dict(argstr='--permuteOrder %s', sep=','), splineGridSize=dict(argstr='--splineGridSize %s', sep=','))
    inputs = BRAINSInitializedControlPoints.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value