from ..tracks import TrackDensityMap
def test_TrackDensityMap_inputs():
    input_map = dict(data_dims=dict(), in_file=dict(extensions=None, mandatory=True), out_filename=dict(extensions=None, usedefault=True), points_space=dict(usedefault=True), reference=dict(extensions=None), voxel_dims=dict())
    inputs = TrackDensityMap.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value