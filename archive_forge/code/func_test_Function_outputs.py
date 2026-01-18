from ..wrappers import Function
def test_Function_outputs():
    output_map = dict()
    outputs = Function.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value