from ..brainsuite import Pvc
def test_Pvc_outputs():
    output_map = dict(outputLabelFile=dict(extensions=None), outputTissueFractionFile=dict(extensions=None))
    outputs = Pvc.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value