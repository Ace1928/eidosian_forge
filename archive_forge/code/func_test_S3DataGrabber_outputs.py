from ..io import S3DataGrabber
def test_S3DataGrabber_outputs():
    output_map = dict()
    outputs = S3DataGrabber.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value