from ..preprocess import AutoTLRC
def test_AutoTLRC_inputs():
    input_map = dict(args=dict(argstr='%s'), base=dict(argstr='-base %s', mandatory=True), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='-input %s', copyfile=False, extensions=None, mandatory=True), no_ss=dict(argstr='-no_ss'), outputtype=dict())
    inputs = AutoTLRC.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value