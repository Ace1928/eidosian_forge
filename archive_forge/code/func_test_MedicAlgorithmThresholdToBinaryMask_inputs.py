from ..developer import MedicAlgorithmThresholdToBinaryMask
def test_MedicAlgorithmThresholdToBinaryMask_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inLabel=dict(argstr='--inLabel %s', sep=';'), inMaximum=dict(argstr='--inMaximum %f'), inMinimum=dict(argstr='--inMinimum %f'), inUse=dict(argstr='--inUse %s'), null=dict(argstr='--null %s'), outBinary=dict(argstr='--outBinary %s', sep=';'), xDefaultMem=dict(argstr='-xDefaultMem %d'), xMaxProcess=dict(argstr='-xMaxProcess %d', usedefault=True), xPrefExt=dict(argstr='--xPrefExt %s'))
    inputs = MedicAlgorithmThresholdToBinaryMask.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value