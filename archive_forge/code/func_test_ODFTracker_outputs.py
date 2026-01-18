from ..odf import ODFTracker
def test_ODFTracker_outputs():
    output_map = dict(track_file=dict(extensions=None))
    outputs = ODFTracker.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value