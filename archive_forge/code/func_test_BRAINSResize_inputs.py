from ..brainsresize import BRAINSResize
def test_BRAINSResize_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputVolume=dict(argstr='--inputVolume %s', extensions=None), outputVolume=dict(argstr='--outputVolume %s', hash_files=False), pixelType=dict(argstr='--pixelType %s'), scaleFactor=dict(argstr='--scaleFactor %f'))
    inputs = BRAINSResize.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value