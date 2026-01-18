from ..registration import LinearRegistration
def test_LinearRegistration_outputs():
    output_map = dict(outputtransform=dict(extensions=None), resampledmovingfilename=dict(extensions=None))
    outputs = LinearRegistration.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value