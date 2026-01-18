from ..segmentation import KellyKapowski
def test_KellyKapowski_outputs():
    output_map = dict(cortical_thickness=dict(extensions=None), warped_white_matter=dict(extensions=None))
    outputs = KellyKapowski.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value