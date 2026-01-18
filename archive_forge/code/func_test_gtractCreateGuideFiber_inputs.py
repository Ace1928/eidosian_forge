from ..gtract import gtractCreateGuideFiber
def test_gtractCreateGuideFiber_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputFiber=dict(argstr='--inputFiber %s', extensions=None), numberOfPoints=dict(argstr='--numberOfPoints %d'), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputFiber=dict(argstr='--outputFiber %s', hash_files=False), writeXMLPolyDataFile=dict(argstr='--writeXMLPolyDataFile '))
    inputs = gtractCreateGuideFiber.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value