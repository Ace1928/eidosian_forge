from ..registration import MultiResolutionAffineRegistration
def test_MultiResolutionAffineRegistration_outputs():
    output_map = dict(resampledImage=dict(extensions=None), saveTransform=dict(extensions=None))
    outputs = MultiResolutionAffineRegistration.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value