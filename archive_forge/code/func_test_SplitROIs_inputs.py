from ..misc import SplitROIs
def test_SplitROIs_inputs():
    input_map = dict(in_file=dict(extensions=None, mandatory=True), in_mask=dict(extensions=None), roi_size=dict())
    inputs = SplitROIs.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value