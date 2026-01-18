from ..specialized import FiducialRegistration
def test_FiducialRegistration_outputs():
    output_map = dict(saveTransform=dict(extensions=None))
    outputs = FiducialRegistration.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value