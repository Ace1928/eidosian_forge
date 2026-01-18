from ..preprocess import Tensor2ApparentDiffusion
def test_Tensor2ApparentDiffusion_outputs():
    output_map = dict(ADC=dict(extensions=None))
    outputs = Tensor2ApparentDiffusion.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value