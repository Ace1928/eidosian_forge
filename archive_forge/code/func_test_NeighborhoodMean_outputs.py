from ..featuredetection import NeighborhoodMean
def test_NeighborhoodMean_outputs():
    output_map = dict(outputVolume=dict(extensions=None))
    outputs = NeighborhoodMean.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value