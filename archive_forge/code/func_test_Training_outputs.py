from ..fix import Training
def test_Training_outputs():
    output_map = dict(trained_wts_file=dict(extensions=None))
    outputs = Training.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value