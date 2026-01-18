from ..developer import MedicAlgorithmSPECTRE2010
def test_MedicAlgorithmSPECTRE2010_outputs():
    output_map = dict(outFANTASM=dict(extensions=None), outMask=dict(extensions=None), outMidsagittal=dict(extensions=None), outOriginal=dict(extensions=None), outPrior=dict(extensions=None), outSegmentation=dict(extensions=None), outSplitHalves=dict(extensions=None), outStripped=dict(extensions=None), outd0=dict(extensions=None))
    outputs = MedicAlgorithmSPECTRE2010.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value