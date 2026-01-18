from ..utils import WarpUtils
def test_WarpUtils_outputs():
    output_map = dict(out_file=dict(extensions=None), out_jacobian=dict(extensions=None))
    outputs = WarpUtils.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value