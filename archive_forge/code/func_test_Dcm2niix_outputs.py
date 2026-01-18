from ..dcm2nii import Dcm2niix
def test_Dcm2niix_outputs():
    output_map = dict(bids=dict(), bvals=dict(), bvecs=dict(), converted_files=dict())
    outputs = Dcm2niix.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value