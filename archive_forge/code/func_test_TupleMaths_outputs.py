from ..maths import TupleMaths
def test_TupleMaths_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = TupleMaths.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value