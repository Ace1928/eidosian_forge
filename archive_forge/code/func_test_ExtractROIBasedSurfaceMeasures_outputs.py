from ..surface import ExtractROIBasedSurfaceMeasures
def test_ExtractROIBasedSurfaceMeasures_outputs():
    output_map = dict(label_files=dict())
    outputs = ExtractROIBasedSurfaceMeasures.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value