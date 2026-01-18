from ..preprocess import DARTELNorm2MNI
def test_DARTELNorm2MNI_outputs():
    output_map = dict(normalization_parameter_file=dict(extensions=None), normalized_files=dict())
    outputs = DARTELNorm2MNI.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value