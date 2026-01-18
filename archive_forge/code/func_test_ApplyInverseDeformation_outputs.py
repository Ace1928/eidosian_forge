from ..utils import ApplyInverseDeformation
def test_ApplyInverseDeformation_outputs():
    output_map = dict(out_files=dict())
    outputs = ApplyInverseDeformation.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value