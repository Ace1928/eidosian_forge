from ..model import FEATRegister
def test_FEATRegister_outputs():
    output_map = dict(fsf_file=dict(extensions=None))
    outputs = FEATRegister.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value