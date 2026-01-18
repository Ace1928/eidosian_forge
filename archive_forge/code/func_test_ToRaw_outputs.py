from ..minc import ToRaw
def test_ToRaw_outputs():
    output_map = dict(output_file=dict(extensions=None))
    outputs = ToRaw.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value