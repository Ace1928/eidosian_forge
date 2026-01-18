from ..preprocess import RobustRegister
def test_RobustRegister_outputs():
    output_map = dict(half_source=dict(extensions=None), half_source_xfm=dict(extensions=None), half_targ=dict(extensions=None), half_targ_xfm=dict(extensions=None), half_weights=dict(extensions=None), out_reg_file=dict(extensions=None), registered_file=dict(extensions=None), weights_file=dict(extensions=None))
    outputs = RobustRegister.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value