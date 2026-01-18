from ..utils import TVtool
def test_TVtool_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='-in %s', extensions=None, mandatory=True), in_flag=dict(argstr='-%s'), out_file=dict(argstr='-out %s', extensions=None, genfile=True))
    inputs = TVtool.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value