from ..brainsuite import Dfs
def test_Dfs_outputs():
    output_map = dict(outputSurfaceFile=dict(extensions=None))
    outputs = Dfs.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value