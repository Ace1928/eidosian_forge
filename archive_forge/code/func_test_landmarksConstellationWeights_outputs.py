from ..brains import landmarksConstellationWeights
def test_landmarksConstellationWeights_outputs():
    output_map = dict(outputWeightsList=dict(extensions=None))
    outputs = landmarksConstellationWeights.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value