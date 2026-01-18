from ..featuredetection import TextureMeasureFilter
def test_TextureMeasureFilter_outputs():
    output_map = dict(outputFilename=dict(extensions=None))
    outputs = TextureMeasureFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value