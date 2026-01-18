from ..preprocess import MRDeGibbs
def test_MRDeGibbs_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = MRDeGibbs.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value