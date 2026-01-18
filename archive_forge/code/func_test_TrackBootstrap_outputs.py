from ..dti import TrackBootstrap
def test_TrackBootstrap_outputs():
    output_map = dict(tracked=dict(extensions=None))
    outputs = TrackBootstrap.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value