from ..base import Merge
def test_Merge_inputs():
    input_map = dict(axis=dict(usedefault=True), no_flatten=dict(usedefault=True), ravel_inputs=dict(usedefault=True))
    inputs = Merge.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value