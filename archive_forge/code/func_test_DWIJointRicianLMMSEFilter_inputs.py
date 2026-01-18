from ..diffusion import DWIJointRicianLMMSEFilter
def test_DWIJointRicianLMMSEFilter_inputs():
    input_map = dict(args=dict(argstr='%s'), compressOutput=dict(argstr='--compressOutput '), environ=dict(nohash=True, usedefault=True), inputVolume=dict(argstr='%s', extensions=None, position=-2), ng=dict(argstr='--ng %d'), outputVolume=dict(argstr='%s', hash_files=False, position=-1), re=dict(argstr='--re %s', sep=','), rf=dict(argstr='--rf %s', sep=','))
    inputs = DWIJointRicianLMMSEFilter.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value