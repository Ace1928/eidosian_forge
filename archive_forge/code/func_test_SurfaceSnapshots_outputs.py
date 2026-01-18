from ..utils import SurfaceSnapshots
def test_SurfaceSnapshots_outputs():
    output_map = dict(snapshots=dict())
    outputs = SurfaceSnapshots.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value