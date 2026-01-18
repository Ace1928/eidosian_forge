from ..preprocess import CANormalize
def test_CANormalize_outputs():
    output_map = dict(control_points=dict(extensions=None), out_file=dict(extensions=None))
    outputs = CANormalize.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value