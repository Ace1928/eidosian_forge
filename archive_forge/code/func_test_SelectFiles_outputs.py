from ..io import SelectFiles
def test_SelectFiles_outputs():
    output_map = dict()
    outputs = SelectFiles.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value