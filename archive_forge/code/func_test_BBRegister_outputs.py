from ..preprocess import BBRegister, BBRegisterInputSpec6
def test_BBRegister_outputs():
    output_map = dict(init_cost_file=dict(), min_cost_file=dict(), out_fsl_file=dict(), out_lta_file=dict(), out_reg_file=dict(), registered_file=dict())
    outputs = BBRegister.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value