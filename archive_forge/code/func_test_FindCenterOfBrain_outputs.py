from ..brains import FindCenterOfBrain
def test_FindCenterOfBrain_outputs():
    output_map = dict(clippedImageMask=dict(extensions=None), debugAfterGridComputationsForegroundImage=dict(extensions=None), debugClippedImageMask=dict(extensions=None), debugDistanceImage=dict(extensions=None), debugGridImage=dict(extensions=None), debugTrimmedImage=dict(extensions=None))
    outputs = FindCenterOfBrain.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value