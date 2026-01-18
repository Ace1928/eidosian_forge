from ..segmentation import JointFusion
def test_JointFusion_outputs():
    output_map = dict(out_atlas_voting_weight=dict(), out_intensity_fusion=dict(), out_label_fusion=dict(extensions=None), out_label_post_prob=dict())
    outputs = JointFusion.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value