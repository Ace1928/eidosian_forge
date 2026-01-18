from ..preprocess import FAST
def test_FAST_outputs():
    output_map = dict(bias_field=dict(), mixeltype=dict(extensions=None), partial_volume_files=dict(), partial_volume_map=dict(extensions=None), probability_maps=dict(), restored_image=dict(), tissue_class_files=dict(), tissue_class_map=dict(extensions=None))
    outputs = FAST.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value