from ..convert import TractShredder
def test_TractShredder_outputs():
    output_map = dict(shredded=dict(extensions=None))
    outputs = TractShredder.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value