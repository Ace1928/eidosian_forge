from ..registration import AffSymTensor3DVol
def test_AffSymTensor3DVol_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = AffSymTensor3DVol.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value