from ..io import JSONFileGrabber
def test_JSONFileGrabber_inputs():
    input_map = dict(defaults=dict(), in_file=dict(extensions=None))
    inputs = JSONFileGrabber.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value