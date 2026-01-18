from ..utils import Tkregister2
def test_Tkregister2_outputs():
    output_map = dict(fsl_file=dict(extensions=None), lta_file=dict(extensions=None), reg_file=dict(extensions=None))
    outputs = Tkregister2.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value