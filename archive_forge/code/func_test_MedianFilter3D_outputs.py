from ..preprocess import MedianFilter3D
def test_MedianFilter3D_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = MedianFilter3D.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value