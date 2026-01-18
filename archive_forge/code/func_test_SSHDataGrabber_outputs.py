from ..io import SSHDataGrabber
def test_SSHDataGrabber_outputs():
    output_map = dict()
    outputs = SSHDataGrabber.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value