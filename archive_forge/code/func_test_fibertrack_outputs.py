from ..fibertrack import fibertrack
def test_fibertrack_outputs():
    output_map = dict(output_fiber_file=dict(extensions=None))
    outputs = fibertrack.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value