from ..metrics import Similarity
def test_Similarity_inputs():
    input_map = dict(mask1=dict(extensions=None), mask2=dict(extensions=None), metric=dict(usedefault=True), volume1=dict(extensions=None, mandatory=True), volume2=dict(extensions=None, mandatory=True))
    inputs = Similarity.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value