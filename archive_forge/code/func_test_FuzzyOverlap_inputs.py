from ..misc import FuzzyOverlap
def test_FuzzyOverlap_inputs():
    input_map = dict(in_mask=dict(extensions=None), in_ref=dict(mandatory=True), in_tst=dict(mandatory=True), out_file=dict(extensions=None, usedefault=True), weighting=dict(usedefault=True))
    inputs = FuzzyOverlap.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value