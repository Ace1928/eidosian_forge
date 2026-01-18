from ..developer import MedicAlgorithmLesionToads
def test_MedicAlgorithmLesionToads_outputs():
    output_map = dict(outCortical=dict(extensions=None), outFilled=dict(extensions=None), outHard=dict(extensions=None), outHard2=dict(extensions=None), outInhomogeneity=dict(extensions=None), outLesion=dict(extensions=None), outMembership=dict(extensions=None), outSulcal=dict(extensions=None), outWM=dict(extensions=None))
    outputs = MedicAlgorithmLesionToads.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value