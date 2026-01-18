from ..registration import Diffeo
def test_Diffeo_outputs():
    output_map = dict(out_file=dict(extensions=None), out_file_xfm=dict(extensions=None))
    outputs = Diffeo.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value