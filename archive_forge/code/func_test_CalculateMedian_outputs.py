from ..misc import CalculateMedian
def test_CalculateMedian_outputs():
    output_map = dict(median_files=dict())
    outputs = CalculateMedian.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value