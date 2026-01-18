from ..brainsuite import Cerebro
def test_Cerebro_outputs():
    output_map = dict(outputAffineTransformFile=dict(extensions=None), outputCerebrumMaskFile=dict(extensions=None), outputLabelVolumeFile=dict(extensions=None), outputWarpTransformFile=dict(extensions=None))
    outputs = Cerebro.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value