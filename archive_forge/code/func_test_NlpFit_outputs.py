from ..minc import NlpFit
def test_NlpFit_outputs():
    output_map = dict(output_grid=dict(extensions=None), output_xfm=dict(extensions=None))
    outputs = NlpFit.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value