from ..maths import MeanImage
def test_MeanImage_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = MeanImage.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value