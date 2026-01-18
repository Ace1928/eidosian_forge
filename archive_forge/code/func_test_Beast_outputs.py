from ..minc import Beast
def test_Beast_outputs():
    output_map = dict(output_file=dict(extensions=None))
    outputs = Beast.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value