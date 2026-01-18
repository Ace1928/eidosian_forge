from ..maths import MathsCommand
def test_MathsCommand_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='%s', extensions=None, mandatory=True, position=2), out_file=dict(argstr='%s', extensions=None, name_source=['in_file'], name_template='%s', position=-2), output_datatype=dict(argstr='-odt %s', position=-3))
    inputs = MathsCommand.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value