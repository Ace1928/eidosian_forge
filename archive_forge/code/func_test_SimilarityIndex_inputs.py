from ..segmentation import SimilarityIndex
def test_SimilarityIndex_inputs():
    input_map = dict(ANNContinuousVolume=dict(argstr='--ANNContinuousVolume %s', extensions=None), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputManualVolume=dict(argstr='--inputManualVolume %s', extensions=None), outputCSVFilename=dict(argstr='--outputCSVFilename %s', extensions=None), thresholdInterval=dict(argstr='--thresholdInterval %f'))
    inputs = SimilarityIndex.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value