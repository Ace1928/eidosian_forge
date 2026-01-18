from ..minc import Copy
def test_Copy_outputs():
    output_map = dict(output_file=dict(extensions=None))
    outputs = Copy.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value