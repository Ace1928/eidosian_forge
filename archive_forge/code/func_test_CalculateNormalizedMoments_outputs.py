from ..misc import CalculateNormalizedMoments
def test_CalculateNormalizedMoments_outputs():
    output_map = dict(moments=dict())
    outputs = CalculateNormalizedMoments.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value