from ..misc import NormalizeProbabilityMapSet
def test_NormalizeProbabilityMapSet_outputs():
    output_map = dict(out_files=dict())
    outputs = NormalizeProbabilityMapSet.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value