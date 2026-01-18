from ..filtering import ResampleScalarVolume
def test_ResampleScalarVolume_inputs():
    input_map = dict(InputVolume=dict(argstr='%s', extensions=None, position=-2), OutputVolume=dict(argstr='%s', hash_files=False, position=-1), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), interpolation=dict(argstr='--interpolation %s'), spacing=dict(argstr='--spacing %s', sep=','))
    inputs = ResampleScalarVolume.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value