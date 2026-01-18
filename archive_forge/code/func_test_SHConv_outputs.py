from ..utils import SHConv
def test_SHConv_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = SHConv.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value