from ..model import EstimateModel
def test_EstimateModel_outputs():
    output_map = dict(ARcoef=dict(), Cbetas=dict(), RPVimage=dict(extensions=['.hdr', '.img', '.img.gz', '.nii']), SDbetas=dict(), SDerror=dict(), beta_images=dict(), labels=dict(extensions=['.hdr', '.img', '.img.gz', '.nii']), mask_image=dict(extensions=['.hdr', '.img', '.img.gz', '.nii']), residual_image=dict(extensions=['.hdr', '.img', '.img.gz', '.nii']), residual_images=dict(), spm_mat_file=dict(extensions=None))
    outputs = EstimateModel.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value