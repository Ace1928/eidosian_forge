from ..registration import ANTS
def test_ANTS_outputs():
    output_map = dict(affine_transform=dict(extensions=None), inverse_warp_transform=dict(extensions=None), metaheader=dict(extensions=None), metaheader_raw=dict(extensions=None), warp_transform=dict(extensions=None))
    outputs = ANTS.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value