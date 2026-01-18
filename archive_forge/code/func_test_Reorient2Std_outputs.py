from ..utils import Reorient2Std
def test_Reorient2Std_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Reorient2Std.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value