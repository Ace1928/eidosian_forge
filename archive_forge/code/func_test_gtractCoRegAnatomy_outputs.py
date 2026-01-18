from ..gtract import gtractCoRegAnatomy
def test_gtractCoRegAnatomy_outputs():
    output_map = dict(outputTransformName=dict(extensions=None))
    outputs = gtractCoRegAnatomy.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value