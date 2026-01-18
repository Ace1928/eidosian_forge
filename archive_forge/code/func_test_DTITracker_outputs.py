from ..dti import DTITracker
def test_DTITracker_outputs():
    output_map = dict(mask_file=dict(extensions=None), track_file=dict(extensions=None))
    outputs = DTITracker.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value