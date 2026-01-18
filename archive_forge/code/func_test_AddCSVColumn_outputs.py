from ..misc import AddCSVColumn
def test_AddCSVColumn_outputs():
    output_map = dict(csv_file=dict(extensions=None))
    outputs = AddCSVColumn.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value