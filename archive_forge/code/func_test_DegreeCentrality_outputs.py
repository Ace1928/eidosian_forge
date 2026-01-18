from ..preprocess import DegreeCentrality
def test_DegreeCentrality_outputs():
    output_map = dict(oned_file=dict(extensions=None), out_file=dict(extensions=None))
    outputs = DegreeCentrality.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value