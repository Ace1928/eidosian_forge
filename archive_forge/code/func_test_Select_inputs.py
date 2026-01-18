from ..base import Select
def test_Select_inputs():
    input_map = dict(index=dict(mandatory=True), inlist=dict(mandatory=True))
    inputs = Select.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value