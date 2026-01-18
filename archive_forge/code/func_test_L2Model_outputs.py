from ..model import L2Model
def test_L2Model_outputs():
    output_map = dict(design_con=dict(extensions=None), design_grp=dict(extensions=None), design_mat=dict(extensions=None))
    outputs = L2Model.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value