from ..arithmetic import SubtractScalarVolumes
def test_SubtractScalarVolumes_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputVolume1=dict(argstr='%s', extensions=None, position=-3), inputVolume2=dict(argstr='%s', extensions=None, position=-2), order=dict(argstr='--order %s'), outputVolume=dict(argstr='%s', hash_files=False, position=-1))
    inputs = SubtractScalarVolumes.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value