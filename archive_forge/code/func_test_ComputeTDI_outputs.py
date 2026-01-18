from ..utils import ComputeTDI
def test_ComputeTDI_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = ComputeTDI.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value