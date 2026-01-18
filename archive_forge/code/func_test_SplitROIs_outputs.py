from ..misc import SplitROIs
def test_SplitROIs_outputs():
    output_map = dict(out_files=dict(), out_index=dict(), out_masks=dict())
    outputs = SplitROIs.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value