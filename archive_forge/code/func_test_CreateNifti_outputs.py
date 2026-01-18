from ..misc import CreateNifti
def test_CreateNifti_outputs():
    output_map = dict(nifti_file=dict(extensions=None))
    outputs = CreateNifti.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value