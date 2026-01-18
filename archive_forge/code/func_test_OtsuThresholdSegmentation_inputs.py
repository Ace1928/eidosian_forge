from ..segmentation import OtsuThresholdSegmentation
def test_OtsuThresholdSegmentation_inputs():
    input_map = dict(args=dict(argstr='%s'), brightObjects=dict(argstr='--brightObjects '), environ=dict(nohash=True, usedefault=True), faceConnected=dict(argstr='--faceConnected '), inputVolume=dict(argstr='%s', extensions=None, position=-2), minimumObjectSize=dict(argstr='--minimumObjectSize %d'), numberOfBins=dict(argstr='--numberOfBins %d'), outputVolume=dict(argstr='%s', hash_files=False, position=-1))
    inputs = OtsuThresholdSegmentation.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value