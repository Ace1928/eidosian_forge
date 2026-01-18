from ..cmtk import CreateNodes
def test_CreateNodes_outputs():
    output_map = dict(node_network=dict(extensions=None))
    outputs = CreateNodes.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value