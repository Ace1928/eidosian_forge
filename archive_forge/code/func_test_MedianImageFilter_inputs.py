from ..denoising import MedianImageFilter
def test_MedianImageFilter_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputVolume=dict(argstr='%s', extensions=None, position=-2), neighborhood=dict(argstr='--neighborhood %s', sep=','), outputVolume=dict(argstr='%s', hash_files=False, position=-1))
    inputs = MedianImageFilter.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value