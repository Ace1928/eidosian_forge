from ..dti import TrackBayesDirac
def test_TrackBayesDirac_outputs():
    output_map = dict(tracked=dict(extensions=None))
    outputs = TrackBayesDirac.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value