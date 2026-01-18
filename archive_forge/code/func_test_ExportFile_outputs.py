from ..io import ExportFile
def test_ExportFile_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = ExportFile.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value