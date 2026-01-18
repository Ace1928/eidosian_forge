from ..vista import Vnifti2Image
def test_Vnifti2Image_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Vnifti2Image.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value