from ..dcm2nii import Dcm2nii
def test_Dcm2nii_outputs():
    output_map = dict(bvals=dict(), bvecs=dict(), converted_files=dict(), reoriented_and_cropped_files=dict(), reoriented_files=dict())
    outputs = Dcm2nii.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value