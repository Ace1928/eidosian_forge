from ..registration import RegistrationSynQuick
def test_RegistrationSynQuick_outputs():
    output_map = dict(forward_warp_field=dict(extensions=None), inverse_warp_field=dict(extensions=None), inverse_warped_image=dict(extensions=None), out_matrix=dict(extensions=None), warped_image=dict(extensions=None))
    outputs = RegistrationSynQuick.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value