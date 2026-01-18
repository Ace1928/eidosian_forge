from ..utils import AverageImages
def test_AverageImages_outputs():
    output_map = dict(output_average_image=dict(extensions=None))
    outputs = AverageImages.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value