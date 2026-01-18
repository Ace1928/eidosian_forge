from ..reg import RegF3D
def test_RegF3D_outputs():
    output_map = dict(avg_output=dict(), cpp_file=dict(extensions=None), invcpp_file=dict(extensions=None), invres_file=dict(extensions=None), res_file=dict(extensions=None))
    outputs = RegF3D.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value