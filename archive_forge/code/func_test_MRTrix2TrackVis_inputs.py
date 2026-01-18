from ..convert import MRTrix2TrackVis
def test_MRTrix2TrackVis_inputs():
    input_map = dict(image_file=dict(extensions=None), in_file=dict(extensions=None, mandatory=True), matrix_file=dict(extensions=None), out_filename=dict(extensions=None, genfile=True, usedefault=True), registration_image_file=dict(extensions=None))
    inputs = MRTrix2TrackVis.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value