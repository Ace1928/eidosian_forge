from ..preprocess import ApplyWarp
def test_ApplyWarp_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = ApplyWarp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value