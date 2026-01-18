from ..registration import AffineRegistration
def test_AffineRegistration_outputs():
    output_map = dict(outputtransform=dict(extensions=None), resampledmovingfilename=dict(extensions=None))
    outputs = AffineRegistration.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value