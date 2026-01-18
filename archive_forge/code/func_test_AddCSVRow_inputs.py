from ..misc import AddCSVRow
def test_AddCSVRow_inputs():
    input_map = dict(_outputs=dict(usedefault=True), in_file=dict(extensions=None, mandatory=True))
    inputs = AddCSVRow.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value