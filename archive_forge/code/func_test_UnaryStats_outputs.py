from ..stats import UnaryStats
def test_UnaryStats_outputs():
    output_map = dict(output=dict())
    outputs = UnaryStats.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value