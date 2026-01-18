from ..misc import FuzzyOverlap
def test_FuzzyOverlap_outputs():
    output_map = dict(class_fdi=dict(), class_fji=dict(), dice=dict(), jaccard=dict())
    outputs = FuzzyOverlap.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value