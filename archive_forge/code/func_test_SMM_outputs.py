from ..model import SMM
def test_SMM_outputs():
    output_map = dict(activation_p_map=dict(extensions=None), deactivation_p_map=dict(extensions=None), null_p_map=dict(extensions=None))
    outputs = SMM.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value