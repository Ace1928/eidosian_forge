from ..model import FEATRegister
def test_FEATRegister_inputs():
    input_map = dict(feat_dirs=dict(mandatory=True), reg_dof=dict(usedefault=True), reg_image=dict(extensions=None, mandatory=True))
    inputs = FEATRegister.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value