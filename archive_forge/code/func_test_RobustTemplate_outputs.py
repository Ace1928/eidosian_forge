from ..longitudinal import RobustTemplate
def test_RobustTemplate_outputs():
    output_map = dict(out_file=dict(extensions=None), scaled_intensity_outputs=dict(), transform_outputs=dict())
    outputs = RobustTemplate.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value