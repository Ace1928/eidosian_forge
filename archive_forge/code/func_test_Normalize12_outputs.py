from ..preprocess import Normalize12
def test_Normalize12_outputs():
    output_map = dict(deformation_field=dict(), normalized_files=dict(), normalized_image=dict())
    outputs = Normalize12.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value