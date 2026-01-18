from ..convert import AnalyzeHeader
def test_AnalyzeHeader_outputs():
    output_map = dict(header=dict(extensions=None))
    outputs = AnalyzeHeader.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value