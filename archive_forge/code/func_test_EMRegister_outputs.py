from ..registration import EMRegister
def test_EMRegister_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = EMRegister.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value