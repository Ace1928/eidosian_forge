from ..converters import DicomToNrrdConverter
def test_DicomToNrrdConverter_outputs():
    output_map = dict(outputDirectory=dict())
    outputs = DicomToNrrdConverter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value