from ..convert import DT2NIfTI
def test_DT2NIfTI_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), header_file=dict(argstr='-header %s', extensions=None, mandatory=True, position=3), in_file=dict(argstr='-inputfile %s', extensions=None, mandatory=True, position=1), output_root=dict(argstr='-outputroot %s', extensions=None, genfile=True, position=2))
    inputs = DT2NIfTI.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value