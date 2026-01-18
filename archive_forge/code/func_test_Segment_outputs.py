from ..preprocess import Segment
def test_Segment_outputs():
    output_map = dict(bias_corrected_image=dict(extensions=None), inverse_transformation_mat=dict(extensions=None), modulated_csf_image=dict(extensions=None), modulated_gm_image=dict(extensions=None), modulated_input_image=dict(deprecated='0.10', extensions=None, new_name='bias_corrected_image'), modulated_wm_image=dict(extensions=None), native_csf_image=dict(extensions=None), native_gm_image=dict(extensions=None), native_wm_image=dict(extensions=None), normalized_csf_image=dict(extensions=None), normalized_gm_image=dict(extensions=None), normalized_wm_image=dict(extensions=None), transformation_mat=dict(extensions=None))
    outputs = Segment.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value