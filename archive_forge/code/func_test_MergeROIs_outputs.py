from ..misc import MergeROIs
def test_MergeROIs_outputs():
    output_map = dict(merged_file=dict(extensions=None))
    outputs = MergeROIs.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value