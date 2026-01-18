from ..reconstruction import CSD
def test_CSD_outputs():
    output_map = dict(model=dict(extensions=None), out_fods=dict(extensions=None))
    outputs = CSD.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value