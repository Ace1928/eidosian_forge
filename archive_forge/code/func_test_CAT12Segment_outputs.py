from ..preprocess import CAT12Segment
def test_CAT12Segment_outputs():
    output_map = dict(bias_corrected_image=dict(extensions=None), csf_dartel_image=dict(extensions=None), csf_modulated_image=dict(extensions=None), csf_native_image=dict(extensions=None), gm_dartel_image=dict(extensions=None), gm_modulated_image=dict(extensions=None), gm_native_image=dict(extensions=None), label_files=dict(), label_roi=dict(extensions=None), label_rois=dict(extensions=None), lh_central_surface=dict(extensions=None), lh_sphere_surface=dict(extensions=None), mri_images=dict(), report=dict(extensions=None), report_files=dict(), rh_central_surface=dict(extensions=None), rh_sphere_surface=dict(extensions=None), surface_files=dict(), wm_dartel_image=dict(extensions=None), wm_modulated_image=dict(extensions=None), wm_native_image=dict(extensions=None))
    outputs = CAT12Segment.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value