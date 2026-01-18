from ..io import DataSink
def test_DataSink_outputs():
    output_map = dict(out_file=dict())
    outputs = DataSink.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value