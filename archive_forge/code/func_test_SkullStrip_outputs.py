from ..preprocess import SkullStrip
def test_SkullStrip_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = SkullStrip.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value