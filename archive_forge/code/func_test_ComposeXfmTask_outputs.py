from ..registration import ComposeXfmTask
def test_ComposeXfmTask_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = ComposeXfmTask.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value