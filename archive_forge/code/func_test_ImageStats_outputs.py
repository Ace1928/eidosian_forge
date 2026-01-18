from ..utils import ImageStats
def test_ImageStats_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = ImageStats.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value