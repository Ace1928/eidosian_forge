from ..tracking import StreamlineTrack
def test_StreamlineTrack_outputs():
    output_map = dict(tracked=dict(extensions=None))
    outputs = StreamlineTrack.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value