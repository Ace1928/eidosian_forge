from ..preprocess import VBMSegment
def test_VBMSegment_outputs():
    output_map = dict(bias_corrected_images=dict(), dartel_input_images=dict(), forward_deformation_field=dict(), inverse_deformation_field=dict(), jacobian_determinant_images=dict(), modulated_class_images=dict(), native_class_images=dict(), normalized_bias_corrected_images=dict(), normalized_class_images=dict(), pve_label_native_images=dict(), pve_label_normalized_images=dict(), pve_label_registered_images=dict(), transformation_mat=dict())
    outputs = VBMSegment.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value