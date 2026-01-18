from ..thresholdscalarvolume import ThresholdScalarVolume
def test_ThresholdScalarVolume_inputs():
    input_map = dict(InputVolume=dict(argstr='%s', extensions=None, position=-2), OutputVolume=dict(argstr='%s', hash_files=False, position=-1), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), lower=dict(argstr='--lower %d'), outsidevalue=dict(argstr='--outsidevalue %d'), threshold=dict(argstr='--threshold %d'), thresholdtype=dict(argstr='--thresholdtype %s'), upper=dict(argstr='--upper %d'))
    inputs = ThresholdScalarVolume.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value