from ..brainsfit import BRAINSFit
def test_BRAINSFit_outputs():
    output_map = dict(bsplineTransform=dict(extensions=None), linearTransform=dict(extensions=None), outputFixedVolumeROI=dict(extensions=None), outputMovingVolumeROI=dict(extensions=None), outputTransform=dict(extensions=None), outputVolume=dict(extensions=None), strippedOutputTransform=dict(extensions=None))
    outputs = BRAINSFit.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value