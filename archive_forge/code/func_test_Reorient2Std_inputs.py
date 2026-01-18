from ..utils import Reorient2Std
def test_Reorient2Std_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='%s', extensions=None, mandatory=True), out_file=dict(argstr='%s', extensions=None, genfile=True, hash_files=False), output_type=dict())
    inputs = Reorient2Std.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value