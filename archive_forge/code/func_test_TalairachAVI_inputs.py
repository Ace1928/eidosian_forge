from ..utils import TalairachAVI
def test_TalairachAVI_inputs():
    input_map = dict(args=dict(argstr='%s'), atlas=dict(argstr='--atlas %s'), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='--i %s', extensions=None, mandatory=True), out_file=dict(argstr='--xfm %s', extensions=None, mandatory=True), subjects_dir=dict())
    inputs = TalairachAVI.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value