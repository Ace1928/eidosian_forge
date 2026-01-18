from ..minc import Voliso
def test_Voliso_inputs():
    input_map = dict(args=dict(argstr='%s'), avgstep=dict(argstr='--avgstep'), clobber=dict(argstr='--clobber', usedefault=True), environ=dict(nohash=True, usedefault=True), input_file=dict(argstr='%s', extensions=None, mandatory=True, position=-2), maxstep=dict(argstr='--maxstep %s'), minstep=dict(argstr='--minstep %s'), output_file=dict(argstr='%s', extensions=None, genfile=True, hash_files=False, name_source=['input_file'], name_template='%s_voliso.mnc', position=-1), verbose=dict(argstr='--verbose'))
    inputs = Voliso.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value