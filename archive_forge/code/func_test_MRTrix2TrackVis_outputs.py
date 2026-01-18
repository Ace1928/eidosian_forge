from ..convert import MRTrix2TrackVis
def test_MRTrix2TrackVis_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = MRTrix2TrackVis.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value