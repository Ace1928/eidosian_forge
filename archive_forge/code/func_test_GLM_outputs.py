from ..model import GLM
def test_GLM_outputs():
    output_map = dict(out_cope=dict(), out_data=dict(), out_f=dict(), out_file=dict(extensions=None), out_p=dict(), out_pf=dict(), out_res=dict(), out_sigsq=dict(), out_t=dict(), out_varcb=dict(), out_vnscales=dict(), out_z=dict())
    outputs = GLM.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value