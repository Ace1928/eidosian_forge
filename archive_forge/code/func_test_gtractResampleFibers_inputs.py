from ..gtract import gtractResampleFibers
def test_gtractResampleFibers_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputForwardDeformationFieldVolume=dict(argstr='--inputForwardDeformationFieldVolume %s', extensions=None), inputReverseDeformationFieldVolume=dict(argstr='--inputReverseDeformationFieldVolume %s', extensions=None), inputTract=dict(argstr='--inputTract %s', extensions=None), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputTract=dict(argstr='--outputTract %s', hash_files=False), writeXMLPolyDataFile=dict(argstr='--writeXMLPolyDataFile '))
    inputs = gtractResampleFibers.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value