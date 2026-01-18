from ..preprocess import RobexSegment
def test_RobexSegment_outputs():
    output_map = dict(out_file=dict(extensions=None), out_mask=dict(extensions=None))
    outputs = RobexSegment.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value