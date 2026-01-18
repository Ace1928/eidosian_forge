from ..postproc import SplineFilter
def test_SplineFilter_outputs():
    output_map = dict(smoothed_track_file=dict(extensions=None))
    outputs = SplineFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value