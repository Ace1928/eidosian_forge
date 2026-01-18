from ..utils import AvScale
def test_AvScale_outputs():
    output_map = dict(average_scaling=dict(), backward_half_transform=dict(), determinant=dict(), forward_half_transform=dict(), left_right_orientation_preserved=dict(), rot_angles=dict(), rotation_translation_matrix=dict(), scales=dict(), skews=dict(), translations=dict())
    outputs = AvScale.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value