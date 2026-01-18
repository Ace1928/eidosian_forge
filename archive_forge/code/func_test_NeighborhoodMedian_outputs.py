from ..featuredetection import NeighborhoodMedian
def test_NeighborhoodMedian_outputs():
    output_map = dict(outputVolume=dict(extensions=None))
    outputs = NeighborhoodMedian.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value