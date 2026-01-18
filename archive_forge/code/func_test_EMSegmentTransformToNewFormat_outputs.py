from ..utilities import EMSegmentTransformToNewFormat
def test_EMSegmentTransformToNewFormat_outputs():
    output_map = dict(outputMRMLFileName=dict(extensions=None))
    outputs = EMSegmentTransformToNewFormat.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value