from ..io import DataGrabber
def test_DataGrabber_outputs():
    output_map = dict()
    outputs = DataGrabber.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value