from ..preprocess import ReplaceFSwithFIRST
def test_ReplaceFSwithFIRST_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = ReplaceFSwithFIRST.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value