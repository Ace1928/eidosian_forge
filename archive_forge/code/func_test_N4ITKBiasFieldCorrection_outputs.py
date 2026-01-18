from ..n4itkbiasfieldcorrection import N4ITKBiasFieldCorrection
def test_N4ITKBiasFieldCorrection_outputs():
    output_map = dict(outputbiasfield=dict(extensions=None), outputimage=dict(extensions=None))
    outputs = N4ITKBiasFieldCorrection.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value