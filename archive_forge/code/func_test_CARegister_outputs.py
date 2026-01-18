from ..preprocess import CARegister
def test_CARegister_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = CARegister.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value