from ..utils import NwarpCat
def test_NwarpCat_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = NwarpCat.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value