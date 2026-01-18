from ..registration import ExpertAutomatedRegistration
def test_ExpertAutomatedRegistration_outputs():
    output_map = dict(resampledImage=dict(extensions=None), saveTransform=dict(extensions=None))
    outputs = ExpertAutomatedRegistration.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value