from ..preprocess import DWIBiasCorrect
def test_DWIBiasCorrect_outputs():
    output_map = dict(bias=dict(extensions=None), out_file=dict(extensions=None))
    outputs = DWIBiasCorrect.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value