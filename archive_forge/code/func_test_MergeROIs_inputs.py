from ..misc import MergeROIs
def test_MergeROIs_inputs():
    input_map = dict(in_files=dict(), in_index=dict(), in_reference=dict(extensions=None))
    inputs = MergeROIs.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value