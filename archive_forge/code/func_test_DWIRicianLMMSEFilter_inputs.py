from ..diffusion import DWIRicianLMMSEFilter
def test_DWIRicianLMMSEFilter_inputs():
    input_map = dict(args=dict(argstr='%s'), compressOutput=dict(argstr='--compressOutput '), environ=dict(nohash=True, usedefault=True), hrf=dict(argstr='--hrf %f'), inputVolume=dict(argstr='%s', extensions=None, position=-2), iter=dict(argstr='--iter %d'), maxnstd=dict(argstr='--maxnstd %d'), minnstd=dict(argstr='--minnstd %d'), mnve=dict(argstr='--mnve %d'), mnvf=dict(argstr='--mnvf %d'), outputVolume=dict(argstr='%s', hash_files=False, position=-1), re=dict(argstr='--re %s', sep=','), rf=dict(argstr='--rf %s', sep=','), uav=dict(argstr='--uav '))
    inputs = DWIRicianLMMSEFilter.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value