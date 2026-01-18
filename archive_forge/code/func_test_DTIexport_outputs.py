from ..diffusion import DTIexport
def test_DTIexport_outputs():
    output_map = dict(outputFile=dict(extensions=None, position=-1))
    outputs = DTIexport.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value