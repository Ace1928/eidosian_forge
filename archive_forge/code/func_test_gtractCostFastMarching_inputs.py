from ..gtract import gtractCostFastMarching
def test_gtractCostFastMarching_inputs():
    input_map = dict(anisotropyWeight=dict(argstr='--anisotropyWeight %f'), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputAnisotropyVolume=dict(argstr='--inputAnisotropyVolume %s', extensions=None), inputStartingSeedsLabelMapVolume=dict(argstr='--inputStartingSeedsLabelMapVolume %s', extensions=None), inputTensorVolume=dict(argstr='--inputTensorVolume %s', extensions=None), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputCostVolume=dict(argstr='--outputCostVolume %s', hash_files=False), outputSpeedVolume=dict(argstr='--outputSpeedVolume %s', hash_files=False), seedThreshold=dict(argstr='--seedThreshold %f'), startingSeedsLabel=dict(argstr='--startingSeedsLabel %d'), stoppingValue=dict(argstr='--stoppingValue %f'))
    inputs = gtractCostFastMarching.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value