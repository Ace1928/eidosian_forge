from ..utils import SmoothTessellation
def test_SmoothTessellation_outputs():
    output_map = dict(surface=dict(extensions=None))
    outputs = SmoothTessellation.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value