from ..developer import JistBrainMgdmSegmentation
def test_JistBrainMgdmSegmentation_outputs():
    output_map = dict(outLevelset=dict(extensions=None), outPosterior2=dict(extensions=None), outPosterior3=dict(extensions=None), outSegmented=dict(extensions=None))
    outputs = JistBrainMgdmSegmentation.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value