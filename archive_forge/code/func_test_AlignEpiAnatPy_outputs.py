from ..preprocess import AlignEpiAnatPy
def test_AlignEpiAnatPy_outputs():
    output_map = dict(anat_al_mat=dict(extensions=None), anat_al_orig=dict(extensions=None), epi_al_mat=dict(extensions=None), epi_al_orig=dict(extensions=None), epi_al_tlrc_mat=dict(extensions=None), epi_reg_al_mat=dict(extensions=None), epi_tlrc_al=dict(extensions=None), epi_vr_al_mat=dict(extensions=None), epi_vr_motion=dict(extensions=None), skullstrip=dict(extensions=None))
    outputs = AlignEpiAnatPy.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value