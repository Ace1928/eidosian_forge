from ..utilities import GenerateEdgeMapImage
def test_GenerateEdgeMapImage_outputs():
    output_map = dict(outputEdgeMap=dict(extensions=None), outputMaximumGradientImage=dict(extensions=None))
    outputs = GenerateEdgeMapImage.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value