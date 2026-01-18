from ..registration import Paint
def test_Paint_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Paint.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value