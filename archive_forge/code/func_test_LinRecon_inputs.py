from ..odf import LinRecon
def test_LinRecon_inputs():
    input_map = dict(args=dict(argstr='%s'), bgmask=dict(argstr='-bgmask %s', extensions=None), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='%s', extensions=None, mandatory=True, position=1), log=dict(argstr='-log'), normalize=dict(argstr='-normalize'), out_file=dict(argstr='> %s', extensions=None, genfile=True, position=-1), qball_mat=dict(argstr='%s', extensions=None, mandatory=True, position=3), scheme_file=dict(argstr='%s', extensions=None, mandatory=True, position=2))
    inputs = LinRecon.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value