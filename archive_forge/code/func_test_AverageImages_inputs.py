from ..utils import AverageImages
def test_AverageImages_inputs():
    input_map = dict(args=dict(argstr='%s'), dimension=dict(argstr='%d', mandatory=True, position=0), environ=dict(nohash=True, usedefault=True), images=dict(argstr='%s', mandatory=True, position=3), normalize=dict(argstr='%d', mandatory=True, position=2), num_threads=dict(nohash=True, usedefault=True), output_average_image=dict(argstr='%s', extensions=None, hash_files=False, position=1, usedefault=True))
    inputs = AverageImages.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value