from ..utils import MRIsCombine
def test_MRIsCombine_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), in_files=dict(argstr='--combinesurfs %s', mandatory=True, position=1), out_file=dict(argstr='%s', extensions=None, genfile=True, mandatory=True, position=-1), subjects_dir=dict())
    inputs = MRIsCombine.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value