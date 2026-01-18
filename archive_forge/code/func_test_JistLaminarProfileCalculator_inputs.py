from ..developer import JistLaminarProfileCalculator
def test_JistLaminarProfileCalculator_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inIntensity=dict(argstr='--inIntensity %s', extensions=None), inMask=dict(argstr='--inMask %s', extensions=None), incomputed=dict(argstr='--incomputed %s'), null=dict(argstr='--null %s'), outResult=dict(argstr='--outResult %s', hash_files=False), xDefaultMem=dict(argstr='-xDefaultMem %d'), xMaxProcess=dict(argstr='-xMaxProcess %d', usedefault=True), xPrefExt=dict(argstr='--xPrefExt %s'))
    inputs = JistLaminarProfileCalculator.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value