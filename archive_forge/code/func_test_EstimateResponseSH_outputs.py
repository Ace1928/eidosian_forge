from ..reconstruction import EstimateResponseSH
def test_EstimateResponseSH_outputs():
    output_map = dict(out_mask=dict(extensions=None), response=dict(extensions=None))
    outputs = EstimateResponseSH.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value