from ..utils import SVAdjustVoxSp
def test_SVAdjustVoxSp_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = SVAdjustVoxSp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value