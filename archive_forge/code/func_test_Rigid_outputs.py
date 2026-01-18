from ..registration import Rigid
def test_Rigid_outputs():
    output_map = dict(out_file=dict(extensions=None), out_file_xfm=dict(extensions=None))
    outputs = Rigid.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value