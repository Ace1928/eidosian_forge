from ..preprocess import FIRST
def test_FIRST_outputs():
    output_map = dict(bvars=dict(), original_segmentations=dict(extensions=None), segmentation_file=dict(extensions=None), vtk_surfaces=dict())
    outputs = FIRST.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value