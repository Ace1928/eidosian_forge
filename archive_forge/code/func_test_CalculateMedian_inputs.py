from ..misc import CalculateMedian
def test_CalculateMedian_inputs():
    input_map = dict(in_files=dict(), median_file=dict(), median_per_file=dict(usedefault=True))
    inputs = CalculateMedian.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value