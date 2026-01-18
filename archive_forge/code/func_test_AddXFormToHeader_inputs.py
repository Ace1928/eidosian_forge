from ..utils import AddXFormToHeader
def test_AddXFormToHeader_inputs():
    input_map = dict(args=dict(argstr='%s'), copy_name=dict(argstr='-c'), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='%s', extensions=None, mandatory=True, position=-2), out_file=dict(argstr='%s', extensions=None, position=-1, usedefault=True), subjects_dir=dict(), transform=dict(argstr='%s', extensions=None, mandatory=True, position=-3), verbose=dict(argstr='-v'))
    inputs = AddXFormToHeader.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value