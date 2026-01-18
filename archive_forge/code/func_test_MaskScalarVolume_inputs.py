from ..arithmetic import MaskScalarVolume
def test_MaskScalarVolume_inputs():
    input_map = dict(InputVolume=dict(argstr='%s', extensions=None, position=-3), MaskVolume=dict(argstr='%s', extensions=None, position=-2), OutputVolume=dict(argstr='%s', hash_files=False, position=-1), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), label=dict(argstr='--label %d'), replace=dict(argstr='--replace %d'))
    inputs = MaskScalarVolume.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value