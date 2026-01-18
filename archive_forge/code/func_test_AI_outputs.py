from ..utils import AI
def test_AI_outputs():
    output_map = dict(output_transform=dict(extensions=None))
    outputs = AI.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value