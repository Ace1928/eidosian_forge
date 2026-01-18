from ..epi import EddyQuad
def test_EddyQuad_outputs():
    output_map = dict(avg_b0_pe_png=dict(), avg_b_png=dict(), clean_volumes=dict(extensions=None), cnr_png=dict(), qc_json=dict(extensions=None), qc_pdf=dict(extensions=None), residuals=dict(extensions=None), vdm_png=dict(extensions=None))
    outputs = EddyQuad.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value