from ..brains import GenerateLabelMapFromProbabilityMap
def test_GenerateLabelMapFromProbabilityMap_outputs():
    output_map = dict(outputLabelVolume=dict(extensions=None))
    outputs = GenerateLabelMapFromProbabilityMap.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value