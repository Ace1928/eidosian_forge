from ..model import MultipleRegressDesign
def test_MultipleRegressDesign_outputs():
    output_map = dict(design_con=dict(extensions=None), design_fts=dict(extensions=None), design_grp=dict(extensions=None), design_mat=dict(extensions=None))
    outputs = MultipleRegressDesign.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value