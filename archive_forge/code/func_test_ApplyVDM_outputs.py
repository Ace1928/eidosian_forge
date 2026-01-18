from ..preprocess import ApplyVDM
def test_ApplyVDM_outputs():
    output_map = dict(mean_image=dict(extensions=None), out_files=dict())
    outputs = ApplyVDM.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value