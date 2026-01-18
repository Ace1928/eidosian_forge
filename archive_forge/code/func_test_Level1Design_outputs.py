from ..model import Level1Design
def test_Level1Design_outputs():
    output_map = dict(spm_mat_file=dict(extensions=None))
    outputs = Level1Design.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value