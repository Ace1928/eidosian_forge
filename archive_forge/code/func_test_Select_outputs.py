from ..base import Select
def test_Select_outputs():
    output_map = dict(out=dict())
    outputs = Select.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value